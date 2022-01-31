from ray.train.api_v2.model import Model
from ray.tune.tune import Tuner
from xgboost_ray.tune import (TuneReportCheckpointCallback,
                              _get_tune_resources)

import ray.data
from ray.train.api_v2.checkpoint import (Checkpoint,
                                         TrainObjectStoreCheckpoint)
from ray.train.api_v2.preprocessor import (Scaler, Chain, Repartitioner)
from ray.train.api_v2.trainer import (ScalingConfig, RunConfig, Trainer)
from sklearn.datasets import load_breast_cancer


@dataclass
class TorchScalingConfig(ScalingConfig):
    num_actors: int
    use_gpu: bool = None
    resources_per_actor: dict = None


class TorchModel(Model):
    def __init__(self, model: nn.Module):
        self.model = model

    def predict(self, dataset: ray.data.Dataset) -> ray.data.Dataset:
        def score_fn(batch):
            # TODO: figure out how to fix this for torch
            return pd.DataFrame(self.model(batch))

        return dataset.map_batches(score_fn, batch_format="pandas")


class TorchTrainer(Trainer):
    def __init__(self,
                 model_cls,
                 train_func,
                 scaling_config: Optional[TorchScalingConfig] = None,
                 run_config: Optional[RunConfig] = None,
                 resume_from_checkpoint: Optional[Checkpoint] = None):
        super().__init__(
            self,
            scaling_config=scaling_config,
            run_config=run_config,
            datasets=None,  # TODO: this shouldn't be here.
            resume_from_checkpoint=resume_from_checkpoint)

        self._trainer = Trainer(
            backend="torch",
            num_workers=scaling_config.num_workers,
            use_gpu=scaling_config.use_gpu,
            resources_per_worker=scaling_config.resources_per_actor)
        self._train_func = train_func

    def as_trainable(self, dataset) -> Type["Trainable"]:
        trainable = self._trainer.to_tune_trainable(self.train_func, dataset)

        def postprocess_checkpoint(config: Dict[str, Any],
                                   checkpoint: Checkpoint):
            checkpoint.preprocessor = config.get("preprocessor", None)

            def _load_model(**options):
                return self.model_fn(checkpoint, **options)

            checkpoint.load_model = _load_model

        trainable.postprocess_checkpoint = postprocess_checkpoint
        return trainable

    def fit(self, dataset: ray.data.Dataset, preprocessor: Preprocessor):
        from ray.tune.tune import Tuner
        tuner = Tuner(
            self,
            run_config=self.run_config,  # I don't know what this does
            param_space={
                "preprocessor": preprocessor,
                **self.kwargs
            })

        result_grid = tuner.fit(datasets={"train_dataset": dataset})
        return result_grid.results[0]

    def model_fn(self, checkpoint: TrainObjectStoreCheckpoint,
                 **options) -> TorchModel:
        # TODO: figure out how to create a checkpoint.
        local_storage_cp = checkpoint.to_local_storage()

        model = ModelCls()
        model.load_state_dict(torch.load(filepath))
        shutil.rmtree(local_storage_cp.path)
        return TorchModel(bst)


def test_xgboost_trainer():
    data_raw = load_breast_cancer(as_frame=True)
    dataset_df = data_raw["data"]
    predict_ds = ray.data.from_pandas(dataset_df)
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

    this_checkpoint = result.checkpoint

    this_model = this_checkpoint.load_model()
    predicted = this_model.predict(predict_ds)
    print(predicted.to_pandas())


test_xgboost_trainer()
