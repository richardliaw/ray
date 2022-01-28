import abc
import copy
from typing import Any, Callable, Dict, Optional, Type, Union

import ray
from ray.data import Dataset
from ray.train.api_v2.model import Model
from ray.train.trainer import wrap_function
from ray.tune.utils.placement_groups import PlacementGroupFactory
from ray.tune.trainable import Trainable
from ray.train.api_v2.checkpoint import (Checkpoint, LocalStorageCheckpoint)
from ray.train.api_v2.preprocessor import Preprocessor

# num_workers, gpu, etc.
ScalingConfig = Dict[str, Any]
# checkpoint_dir, etc.
RunConfig = Dict[str, Any]

# dataset / dataset factory
GenDataset = Union[Dataset, Callable[[], Dataset]]


class ConvertibleToTrainable(abc.ABC):
    def as_trainable(self) -> Type["Trainable"]:
        raise NotImplementedError


class Trainer(ConvertibleToTrainable, abc.ABC):
    def __init__(self,
                 scaling_config: Optional[ScalingConfig] = None,
                 run_config: Optional[RunConfig] = None,
                 datasets: Optional[dict] = None,
                 resume_from_checkpoint: Optional[Checkpoint] = None,
                 **kwargs):
        self.scaling_config = scaling_config
        self.run_config = run_config
        self.datasets = datasets
        self.resume_from_checkpoint = resume_from_checkpoint
        self.kwargs = kwargs

    def fit(self, dataset: ray.data.Dataset, preprocessor: Preprocessor):
        raise NotImplementedError


class FunctionTrainer(Trainer, abc.ABC):
    def train_fn(self, run_config: RunConfig, scaling_config: ScalingConfig,
                 datasets: Dict[str, ray.data.Dataset],
                 checkpoint: Optional[Checkpoint], **kwargs) -> Any:
        raise NotImplementedError

    def resource_fn(self,
                    scaling_config: ScalingConfig) -> PlacementGroupFactory:
        raise NotImplementedError

    def model_fn(self, checkpoint: Checkpoint, **options) -> Model:
        raise NotImplementedError

    def fit(self, dataset: ray.data.Dataset, preprocessor: Preprocessor):
        from ray.tune.tune import Tuner

        trainable = self.as_trainable()

        tuner = Tuner(
            trainable,
            run_config=self.run_config,
            param_space={
                "preprocessor": preprocessor,
                **self.kwargs
            })
        result_grid = tuner.fit(datasets={"train_dataset": dataset})
        return result_grid.results[0]

    def as_trainable(self) -> Type["Trainable"]:
        self_run_config = copy.deepcopy(self.run_config) or {}
        self_scaling_config = copy.deepcopy(self.scaling_config) or {}
        self_datasets = copy.copy(self.datasets)
        self_resume_from_checkpoint = copy.deepcopy(
            self.resume_from_checkpoint)
        self_kwargs = copy.deepcopy(self.kwargs)

        # Using a function trainable here as XGBoost-Ray's integrations
        # (e.g. callbacks) are optimized for this case
        def internal_train_fn(config, checkpoint_dir):
            override_run_config = config.pop("run_config", None)
            override_scaling_config = config.pop("scaling_config", None)

            datasets = self_datasets or {}
            override_datasets = config.pop("datasets", None)
            if override_datasets:
                datasets.update(override_datasets)

            preprocessor = config.pop("preprocessor", None)

            if preprocessor:
                processed_datasets = {
                    name: preprocessor.fit_transform(ds)
                    for name, ds in datasets.items()
                }
            else:
                processed_datasets = datasets

            run_config = self_run_config or {}
            if override_run_config:
                run_config.update(override_run_config)

            scaling_config = self_scaling_config or {}
            if override_scaling_config:
                scaling_config.update(override_scaling_config)

            def update(name: str, arg: Any, config: Dict[str, Any]):
                # Pop config items to allow fo deep override
                config_item = config.pop(name, None)
                if isinstance(arg, dict) and isinstance(config_item, dict):
                    arg.update(config_item)
                    return arg
                return config_item or arg

            updated_kwargs = {
                name: update(name, arg, config)
                for name, arg in self_kwargs.items()
            }

            # Update with remaining config ite s
            updated_kwargs.update(config)

            if checkpoint_dir:
                checkpoint = LocalStorageCheckpoint(path=checkpoint_dir)
            else:
                checkpoint = self_resume_from_checkpoint

            self.train_fn(
                run_config=run_config,
                scaling_config=scaling_config,
                datasets=processed_datasets,
                checkpoint=checkpoint,
                **updated_kwargs)

        trainable = wrap_function(internal_train_fn)
        trainable.__name__ = self.train_fn.__name__

        # Monkey patching the resource requests for dynamic resource
        # allocation
        def resource_request(config):
            config = copy.deepcopy(config)

            scaling_config = {
                "num_actors": 0,
                "cpus_per_actor": 1,
                "gpus_per_actor": 1,
                "resources_per_actor": None
            }
            if self_scaling_config:
                scaling_config.update(self_scaling_config)

            override_scaling_config = config.pop("scaling_config", None)
            if override_scaling_config:
                scaling_config.update(override_scaling_config)

            return self.resource_fn(scaling_config)

        trainable.default_resource_request = resource_request

        def postprocess_checkpoint(config: Dict[str, Any],
                                   checkpoint: Checkpoint):
            checkpoint.preprocessor = config.get("preprocessor", None)

            def _load_model(**options):
                return self.model_fn(checkpoint, **options)

            checkpoint.load_model = _load_model

        trainable.postprocess_checkpoint = postprocess_checkpoint

        return trainable


def trainable(train_fn: Callable[[
        RunConfig, ScalingConfig, Dict[str, ray.data.Dataset], Optional[
            Checkpoint], Any
], Any], resource_fn: Callable[[ScalingConfig], PlacementGroupFactory],
              model_fn: Callable[[Checkpoint], Model]):
    class TuneTrainer(FunctionTrainer):
        def train_fn(self, run_config: RunConfig,
                     scaling_config: ScalingConfig,
                     datasets: Dict[str, ray.data.Dataset],
                     checkpoint: Optional[Checkpoint], **kwargs):
            return train_fn(run_config, scaling_config, datasets, checkpoint,
                            **kwargs)

        def resource_fn(self, scaling_config: ScalingConfig):
            return resource_fn(scaling_config)

        def model_fn(self, checkpoint: Checkpoint, **options) -> Model:
            return model_fn(checkpoint)

    return TuneTrainer
