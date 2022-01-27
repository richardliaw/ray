import os
from typing import Type, Optional

import xgboost_ray
from ray import tune
from ray.tune.function_runner import wrap_function
from xgboost_ray.tune import TuneReportCheckpointCallback, _TuneCheckpointCallback

import ray.data
from ray.train.api_v2.checkpoint import Checkpoint
from ray.train.api_v2.preprocessor import (Preprocessor, Scaler, Chain,
                                           Repartitioner)
from ray.train.api_v2.trainer import Result, Trainer, ScalingConfig, RunConfig
from ray.tune.trainable import Trainable
# from ray.tune.tune import Tuner
from sklearn.datasets import load_breast_cancer


class XGBoostResult(Result):
    pass


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
        self.params = params or {"objective": "binary:logistic"}
        self.ray_params = ray_params
        self.xgboost_args = xgboost_args
        self.xgboost_kwargs = xgboost_kwargs
        self.label = label

    def fit(self, dataset: ray.data.Dataset,
            preprocessor: Preprocessor) -> Result:
        processed = preprocessor.fit_transform(dataset)

        dmatrix = xgboost_ray.RayDMatrix(processed, label=self.label)
        evals_result = {}

        # This is only needed for the new `model_file` parameter.
        # Upstream this eventually
        class _TuneCheckpointModelCallback(_TuneCheckpointCallback):
            @staticmethod
            def _create_checkpoint(model, epoch: int, filename: str,
                                   frequency: int):
                if epoch % frequency > 0:
                    return
                with tune.checkpoint_dir(
                        step=epoch, model_file=filename) as checkpoint_dir:
                    model.save_model(os.path.join(checkpoint_dir, filename))

        class TuneReportCheckpointModelCallback(TuneReportCheckpointCallback):
            _checkpoint_callback_cls = _TuneCheckpointModelCallback

        ray_params = xgboost_ray.RayParams()
        ray_params.__dict__.update(**self.run_config)
        ray_params.__dict__.update(**self.scaling_config)

        bst = xgboost_ray.train(
            dtrain=dmatrix,
            params=self.params,
            evals_result=evals_result,
            ray_params=ray_params,
            # callbacks=[TuneReportCheckpointModelCallback(
            #   filename="model.xgb", frequency=1)],
            *self.xgboost_args,
            **self.xgboost_kwargs)

        return XGBoostResult(metrics=evals_result, checkpoint=bst)

    def as_trainable(self) -> Type["Trainable"]:
        def SimpleRayXGBoostTrainable(config):
            pass

        return wrap_function(SimpleRayXGBoostTrainable)


def test_xgboost_trainer():
    data_raw = load_breast_cancer(as_frame=True)
    dataset_df = data_raw["data"]
    dataset_df["target"] = data_raw["target"]
    dataset = ray.data.from_pandas(dataset_df)

    preprocessor = Chain(
        Scaler(["worst radius", "worst area"]),
        Repartitioner(num_partitions=2))

    config = {
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
        params=config)
    result = trainer.fit(dataset, preprocessor=preprocessor)
    print(result)


test_xgboost_trainer()

#
# tuner = Tuner(
#
# )
