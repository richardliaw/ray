import abc
from typing import Any, Callable, Dict, Optional, Type, Union

import ray
from ray.data import Dataset
from ray.tune.trainable import Trainable
from ray.train.api_v2.checkpoint import Checkpoint
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
                 scaling_config: ScalingConfig,
                 run_config: RunConfig,
                 datasets: Optional[dict] = None,
                 resume_from_checkpoint: Optional[Checkpoint] = None,
                 **kwargs):
        self.scaling_config = scaling_config
        self.run_config = run_config
        self.datasets = datasets
        self.resume_from_checkpoint = resume_from_checkpoint

    def fit(self, dataset: ray.data.Dataset, preprocessor: Preprocessor):
        raise NotImplementedError


class Result:
    def __init__(self, metrics: Dict[str, Any], checkpoint: Any):
        self.metrics = metrics
        self.checkpoint = checkpoint
