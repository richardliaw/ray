from typing import Optional, Dict, Any

import xgboost_ray
from ray import tune
from ray.tune.tune import Tuner
from xgboost_ray.tune import (TuneReportCheckpointCallback,
                              _get_tune_resources)

import ray.data
from ray.train.api_v2.checkpoint import Checkpoint
from ray.train.api_v2.preprocessor import (Scaler, Chain, Repartitioner)
from ray.train.api_v2.trainer import (ScalingConfig, RunConfig,
                                      FunctionTrainer)
from sklearn.datasets import load_breast_cancer


class XGBoostTrainer(FunctionTrainer):
    def train_fn(self, run_config: RunConfig, scaling_config: ScalingConfig,
                 datasets: Dict[str, ray.data.Dataset],
                 checkpoint: Optional[Checkpoint], label: str,
                 params: Optional[Dict[str, Any]], **kwargs):
        train_dataset = datasets["train_dataset"]

        dmatrix = xgboost_ray.RayDMatrix(train_dataset, label=label)
        evals_result = {}

        ray_params = xgboost_ray.RayParams()
        ray_params.__dict__.update(**run_config)
        ray_params.__dict__.update(**scaling_config)

        xgboost_ray.train(
            dtrain=dmatrix,
            params=params,
            evals_result=evals_result,
            ray_params=ray_params,
            callbacks=[
                TuneReportCheckpointCallback(
                    filename="model.xgb", frequency=1)
            ],
            **kwargs)

    def resource_fn(self, scaling_config: ScalingConfig):
        return _get_tune_resources(**scaling_config)


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

    trainer = XGBoostTrainer(
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
        XGBoostTrainer(
            run_config={"max_actor_restarts": 1},
            scaling_config={},
            datasets={},
            resume_from_checkpoint=None,
            label="target"),
        run_config={},
        param_space=param_space)

    results = tuner.fit()
    print(results.results)

    best_result = results.results[0]
    best_checkpoint = best_result.checkpoint
    print(best_result.metrics, best_checkpoint)


# test_xgboost_trainer()
test_xgboost_tuner()

#
# tuner = Tuner(
#
# )
