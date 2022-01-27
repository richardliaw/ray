import copy
from typing import Type, Optional

import xgboost_ray
from ray import tune
from ray.train.api_v2.result import Result
from ray.tune.tune import Tuner
from ray.tune.function_runner import wrap_function
from xgboost_ray.tune import (TuneReportCheckpointCallback,
                              _get_tune_resources)

import ray.data
from ray.train.api_v2.checkpoint import Checkpoint
from ray.train.api_v2.preprocessor import (Preprocessor, Scaler, Chain,
                                           Repartitioner)
from ray.train.api_v2.trainer import Trainer, ScalingConfig, RunConfig
from ray.tune.trainable import Trainable
from sklearn.datasets import load_breast_cancer


class SimpleRayXGBoostTrainer(Trainer):
    def __init__(self,
                 scaling_config: ScalingConfig,
                 run_config: RunConfig,
                 datasets: Optional[dict] = None,
                 resume_from_checkpoint: Optional[Checkpoint] = None,
                 label: Optional[str] = None,
                 params: Optional[dict] = None,
                 ray_params: Optional[xgboost_ray.RayParams] = None,
                 *xgboost_args,
                 **xgboost_kwargs):
        super(SimpleRayXGBoostTrainer, self).__init__(
            scaling_config=scaling_config,
            run_config=run_config,
            datasets=datasets,
            resume_from_checkpoint=resume_from_checkpoint)
        self.label = label
        self.params = params
        self.ray_params = ray_params
        self.xgboost_args = xgboost_args
        self.xgboost_kwargs = xgboost_kwargs

    def fit(self,
            train_dataset: ray.data.Dataset,
            preprocessor: Optional[Preprocessor] = None) -> Result:
        trainable = self.as_trainable()

        tuner = Tuner(
            trainable,
            run_config=self.run_config,
            param_space={
                "datasets": {
                    "train_dataset": train_dataset
                },
                "preprocessor": preprocessor,
                "ray_params": self.ray_params,
                **self.params
            })
        result_grid = tuner.fit()
        return result_grid.results[0]

    def as_trainable(self) -> Type["Trainable"]:
        self_run_config = copy.deepcopy(self.run_config)
        self_scaling_config = copy.deepcopy(self.scaling_config)
        self_params = copy.deepcopy(self.params)
        self_label = self.label
        self_xgboost_kwargs = copy.deepcopy(self.xgboost_kwargs)

        # Using a function trainable here as XGBoost-Ray's integrations
        # (e.g. callbacks) are optimized for this case
        def SimpleRayXGBoostTrainable(config):
            override_run_config = config.pop("run_config", None)
            override_scaling_config = config.pop("scaling_config", None)

            datasets = config.pop("datasets", None)
            preprocessor = config.pop("preprocessor", None)

            run_config = self_run_config or {}
            if override_run_config:
                run_config.update(override_run_config)

            scaling_config = self_scaling_config or {}
            if override_scaling_config:
                scaling_config.update(override_scaling_config)

            train_dataset = datasets["train_dataset"]

            processed = preprocessor.fit_transform(train_dataset)

            label = config.pop("label")

            dmatrix = xgboost_ray.RayDMatrix(
                processed, label=label or self_label)
            evals_result = {}

            ray_params = xgboost_ray.RayParams()
            ray_params.__dict__.update(**run_config)
            ray_params.__dict__.update(**scaling_config)

            params = self_params or {}
            override_params = config.pop("params")
            if override_params:
                params.update(override_params)

            xgboost_args = self.xgboost_args
            xgboost_kwargs = self_xgboost_kwargs
            xgboost_kwargs.update(config)

            xgboost_ray.train(
                dtrain=dmatrix,
                params=params,
                evals_result=evals_result,
                ray_params=ray_params,
                callbacks=[
                    TuneReportCheckpointCallback(
                        filename="model.xgb", frequency=1)
                ],
                *xgboost_args,
                **xgboost_kwargs)

        trainable = wrap_function(SimpleRayXGBoostTrainable)

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

            return _get_tune_resources(**scaling_config)

        trainable.default_resource_request = resource_request
        return trainable


# DRAFT: Resource/Model API
#
# class TrainableClass(Trainable):
#     def step(self):
#         pass
#
#     def default_resource_request(cls, config: Dict[str, Any]) -> \
#             Union[Resources, PlacementGroupFactory]:
#         pass
#
#     def save_checkpoint(self, tmp_checkpoint_dir):
#         pass
#
#     def export_model(self, export_formats, export_dir=None):
#         pass
#
#     def get_model_from_checkpoint(self, checkpoint):
#         pass
#
#
# def train_fn(config):
#     pass
#
#
# def resource_fn(config) -> PlacementGroupFactory:
#     pass
#
#
# def model_fn(checkpoint) -> Model:
#     pass
#
#
# tune.run(tune.trainable(train_fn, resource_fn, model_fn), )


def test_xgboost_trainer():
    data_raw = load_breast_cancer(as_frame=True)
    dataset_df = data_raw["data"]
    dataset_df["target"] = data_raw["target"]
    dataset = ray.data.from_pandas(dataset_df)

    preprocessor = Chain(
        Scaler(["worst radius", "worst area"]),
        Repartitioner(num_partitions=2))

    params = {
        "tree_method": "approx",
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
    }

    trainer = SimpleRayXGBoostTrainer(
        scaling_config={
            "num_actors": 2,
            "gpus_per_actor": 0,
            "cpus_per_actor": 2,
        },
        run_config={"max_actor_restarts": 1},
        label="target",
        params=params)
    result = trainer.fit(dataset, preprocessor=preprocessor)
    print(result)


# test_xgboost_trainer()


def test_xgboost_tuner():
    data_raw = load_breast_cancer(as_frame=True)
    dataset_df = data_raw["data"]
    dataset_df["target"] = data_raw["target"]
    dataset = ray.data.from_pandas(dataset_df)

    # Tune datasets
    dataset_v1 = dataset.random_shuffle(seed=1234)
    dataset_v2, _ = dataset.random_shuffle(seed=2345).split(2, equal=True)

    dataset_v1.get_internal_block_refs()
    dataset_v2.get_internal_block_refs()

    # For Tune table output (makes it easier to debug)
    ray.data.Dataset.__repr__ = lambda self: (f"<Dataset num_rows="
                                              f"{self._meta_count()}>")

    # Tune preprocessors
    prep_v1 = Chain(
        Scaler(["worst radius", "worst area"]),
        Repartitioner(num_partitions=4))

    prep_v2 = Chain(
        Scaler(["worst concavity", "worst smoothness"]),
        Repartitioner(num_partitions=8))

    param_space = {
        "scaling_config": {
            "num_actors": tune.grid_search([2, 4]),
            "cpus_per_actor": 2,
            "gpus_per_actor": 0
        },
        "preprocessor": tune.grid_search([prep_v1, prep_v2]),
        "datasets": {
            "train_dataset": tune.grid_search([dataset_v1, dataset_v2]),
        },
        "label": "target",
        "params": {
            "objective": "binary:logistic",
            "tree_method": "approx",
            "eval_metric": ["logloss", "error"],
            "eta": tune.loguniform(1e-4, 1e-1),
            "subsample": tune.uniform(0.5, 1.0),
            "max_depth": tune.randint(1, 9)
        }
    }

    tuner = Tuner(
        SimpleRayXGBoostTrainer,
        run_config={"max_actor_restarts": 1},
        param_space=param_space)

    results = tuner.fit()
    print(results.results)

    best_result = results.results[0]
    best_checkpoint = best_result.checkpoint
    print(best_result.metrics, best_checkpoint)


test_xgboost_tuner()

#
# tuner = Tuner(
#
# )
