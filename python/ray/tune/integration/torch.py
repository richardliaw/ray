# Original Code here:
# https://github.com/pytorch/examples/blob/master/mnist/main.py
from contextlib import contextmanager
import os
from functools import partial
import logging
import shutil
import tempfile
from typing import Callable, Dict, Generator, Optional, Type

import torch
from datetime import timedelta
from dataclasses import dataclass

import ray
from ray import tune
from ray.tune.result import RESULT_DUPLICATE
from ray.tune.logger import NoopLogger
from ray.tune.function_runner import wrap_function
from ray.tune.resources import Resources
from ray.tune.trainable import TrainableUtil
from ray.tune.utils import detect_checkpoint_function
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.sgd.torch.utils import setup_process_group, setup_address
from ray.util.sgd.torch.constants import NCCL_TIMEOUT_S

logger = logging.getLogger(__name__)

_distributed_enabled = False


@dataclass
class ResourceConfig:
    """
        cpus_per_worker (int): Number of CPU resources to reserve
            per training worker.
    """
    workers_per_host: Optional[int] = None
    cpus_per_worker: int = 1
    gpus_per_worker: int = None


def is_distributed_trainable():
    """Returns True if executing within a DistributedTrainable."""
    return _distributed_enabled


def enable_distributed_trainable():
    global _distributed_enabled
    _distributed_enabled = True


def logger_creator(log_config: Dict, logdir: str, rank: int) -> NoopLogger:
    worker_dir = os.path.join(logdir, "worker_{}".format(rank))
    os.makedirs(worker_dir, exist_ok=True)
    return NoopLogger(log_config, worker_dir)


class _TorchTrainable(tune.Trainable):
    """Base class for distributed training on Tune.

    A wrapper class is needed to actually create a working
    version of this trainable.
    """
    _function = None
    _num_workers = None
    _use_gpu = None
    _resource_config = None
    _placement_group = None

    __slots__ = ["workers", "_finished", "_placement_group"]

    @classmethod
    def default_process_group_parameters(self) -> Dict:
        return dict(timeout=timedelta(NCCL_TIMEOUT_S), backend="gloo")

    @property
    def worker_gpus(self) -> int:
        if self._use_gpu:
            return 0
        return self._resource_config.gpus_per_worker

    @property
    def worker_cpus(self) -> int:
        return self._resource_config.cpus_per_worker

    @property
    def should_colocate(self) -> bool:
        return bool(self._resource_config.workers_per_host)

    @property
    def num_hosts(self) -> Optional[int]:
        if self.should_colocate:
            return int(
                self._num_workers / self._resource_config.workers_per_host)

    def get_remote_worker_options(self) -> Dict[str, int]:
        options = dict(num_cpus=self.worker_cpus, num_gpus=self.worker_gpus)
        if self.should_colocate:
            bundle = {
                "CPU": self.worker_cpus * self.workers_per_host,
                "GPU": self.worker_gpus * self.workers_per_host,
            }
            all_bundles = [bundle] * self.num_hosts
            self._placement_group = placement_group(
                all_bundles, strategy="STRICT_SPREAD")
            logger.debug("Waiting for placement_group to start.")
            ray.get(self._placement_group.ready())
            logger.debug("Placement_group started.")
            options["placement_group"] = self.placement_group

        return options

    def setup(self, config: Dict):
        self._finished = False
        num_workers = self._num_workers
        logdir = self.logdir
        assert self._function

        func_trainable = wrap_function(self.__class__._function)

        remote_trainable = ray.remote(func_trainable)
        remote_trainable = remote_trainable.options(
            **self.get_remote_worker_options())

        self.workers = [
            remote_trainable.remote(
                config=config,
                logger_creator=lambda cfg: logger_creator(cfg, logdir, rank))
            for rank in range(num_workers)
        ]

        # Address has to be IP of rank 0 worker's node.
        address = ray.get(
            self.workers[0].execute.remote(lambda _: setup_address()))

        pgroup_params = self.default_process_group_parameters()

        setup_on_worker = partial(
            setup_process_group,
            url=address,
            world_size=num_workers,
            **pgroup_params)
        ray.get([
            w.execute.remote(lambda _: setup_on_worker(world_rank=rank))
            for rank, w in enumerate(self.workers)
        ])

        ray.get([
            w.execute.remote(lambda _: enable_distributed_trainable())
            for rank, w in enumerate(self.workers)
        ])

    def step(self) -> Dict:
        if self._finished:
            raise RuntimeError("Training has already finished.")
        result = ray.get([w.step.remote() for w in self.workers])[0]
        if RESULT_DUPLICATE in result:
            self._finished = True
        return result

    def save_checkpoint(self, checkpoint_dir: str) -> str:
        # TODO: optimize if colocated
        save_obj = ray.get(self.workers[0].save_to_object.remote())
        checkpoint_path = TrainableUtil.create_from_pickle(
            save_obj, checkpoint_dir)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_dir: str):
        checkpoint_obj = TrainableUtil.checkpoint_to_object(checkpoint_dir)
        return ray.get(
            w.restore_from_object.remote(checkpoint_obj) for w in self.workers)

    def stop(self):
        ray.get([worker.stop.remote() for worker in self.workers])
        if self.should_colocate:
            remove_placement_group(self._placement_group)


def DistributedTrainableCreator(
        func: Callable,
        use_gpu: bool = False,
        num_workers: int = 1,
        resource_config: ResourceConfig = None,
        backend: str = "gloo",
        timeout_s: int = NCCL_TIMEOUT_S) -> Type[_TorchTrainable]:
    """Creates a class that executes distributed training.

    Similar to running `torch.distributed.launch`.

    Note that you typically should not instantiate the object
    created.

    Args:
        func (callable): This function is a Tune trainable function.
            This function must have 2 args in the signature, and the
            latter arg must contain `checkpoint_dir`. For example:
            `func(config, checkpoint_dir=None)`.
        use_gpu (bool): Sets resource allocation for workers to 1 GPU
            if true. Also automatically sets CUDA_VISIBLE_DEVICES
            for each training worker.
        num_workers (int): Number of training workers to include in
            world.
        backend (str): One of "gloo", "nccl".
        timeout_s (float): Seconds before the torch process group
            times out. Useful when machines are unreliable. Defaults
            to 60 seconds.

    Returns:
        A trainable class object that can be passed to Tune. Resources
            are automatically set within the object, so users do
            not need to set `resources_per_trainable`.

    Example:

    .. code-block:: python

        trainable_cls = DistributedTrainableCreator(
            train_func, num_workers=2)
        analysis = tune.run(trainable_cls)
    """
    detect_checkpoint_function(func, abort=True)
    resource_config = resource_config or ResourceConfig()
    if resource_config.workers_per_host:
        if num_workers % resource_config.workers_per_host:
            raise ValueError("`num_workers` must be an integer multiple "
                             "of resource_config.workers_per_host.")
    if not use_gpu and resource_config.gpus_per_worker:
        raise ValueError("use_gpu must be set if `resource_config."
                         "gpus_per_worker` is provided.")

    class WrappedDistributedTorchTrainable(_TorchTrainable):
        _function = func
        _num_workers = num_workers
        _use_gpu = use_gpu
        _resource_config = resource_config

        @classmethod
        def default_process_group_parameters(self) -> Dict:
            return dict(timeout=timedelta(timeout_s), backend=backend)

        @classmethod
        def default_resource_request(cls, config: Dict) -> Resources:
            cpus_per_worker = resource_config.cpus_per_worker
            gpus_per_worker = resource_config.gpus_per_worker

            num_workers_ = int(config.get("num_workers", num_workers))
            use_gpu_ = config.get("use_gpu", use_gpu)

            return Resources(
                cpu=0,
                gpu=0,
                extra_cpu=cpus_per_worker * num_workers_,
                extra_gpu=num_workers_ * gpus_per_worker if use_gpu_ else 0)

    return WrappedDistributedTorchTrainable


@contextmanager
def distributed_checkpoint_dir(
        step: int, disable: bool = False) -> Generator[str, None, None]:
    """ContextManager for creating a distributed checkpoint.

    Only checkpoints a file on the "main" training actor, avoiding
    redundant work.

    Args:
        step (int): Used to label the checkpoint
        disable (bool): Disable for prototyping.

    Yields:
        path (str): A path to a directory. This path will be used
            again when invoking the training_function.
    Example:

    .. code-block:: python

        def train_func(config, checkpoint_dir):
            if checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                model_state_dict = torch.load(path)

            if epoch % 3 == 0:
                with distributed_checkpoint_dir(step=epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save(model.state_dict(), path)
    """

    if torch.distributed.get_rank() == 0 and not disable:
        with tune.checkpoint_dir(step=step) as checkpoint_dir:
            yield checkpoint_dir
    else:
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path)


def _train_check_global(config: Dict, checkpoint_dir: Optional[str] = None):
    """For testing only. Putting this here because Ray has problems
    serializing within the test file."""
    assert is_distributed_trainable()
    import time
    time.sleep(0.1)
    tune.report(is_distributed=True)


def _train_simple(config: Dict, checkpoint_dir: Optional[str] = None):
    """For testing only. Putting this here because Ray has problems
    serializing within the test file."""
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel
    import torch.optim as optim
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 8, 5, 5, 5

    # Create random Tensors to hold inputs and outputs
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)
    loss_fn = nn.MSELoss()

    # Use the nn package to define our model and loss function.
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    )
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    if checkpoint_dir:
        with open(os.path.join(checkpoint_dir, "checkpoint")) as f:
            model_state, optimizer_state = torch.load(f)

        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    model = DistributedDataParallel(model)

    for epoch in range(config.get("epochs", 10)):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        if epoch % 3 == 0:
            if config.get("enable_checkpoint", True):
                with distributed_checkpoint_dir(step=epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((model.state_dict(), optimizer.state_dict()),
                               path)
        tune.report(mean_loss=loss.item())
