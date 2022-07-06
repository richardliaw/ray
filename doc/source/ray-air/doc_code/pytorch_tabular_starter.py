# isort: skip_file

# __air_pytorch_preprocess_start__
import numpy as np
import ray
from ray.data.preprocessors import StandardScaler, BatchMapper, Chain
from ray.air import train_test_split
import pandas as pd

# Load data.
dataset = ray.data.read_csv("s3://air-example-data/breast_cancer.csv")
# Split data into train and validation.
train_dataset, valid_dataset = train_test_split(dataset, test_size=0.3)

# Create a test dataset by dropping the target column.
test_dataset = valid_dataset.map_batches(
    lambda df: df.drop("target", axis=1), batch_format="pandas"
)

# Get the training data schema
schema_order = [k for k in train_dataset.schema().names if k != "target"]


def concat_for_tensor(dataframe):
    # Concatenate the dataframe into a single tensor.
    from ray.data.extensions import TensorArray

    result = {}
    result["input"] = TensorArray(dataframe[schema_order].to_numpy(dtype=np.float32))
    if "target" in dataframe:
        result["target"] = TensorArray(dataframe["target"].to_numpy(dtype=np.float32))
    return pd.DataFrame(result)


# Create a preprocessor to scale some columns
columns_to_scale = ["mean radius", "mean texture"]

preprocessor = Chain(
    StandardScaler(columns=columns_to_scale), BatchMapper(concat_for_tensor)
)
# __air_pytorch_preprocess_end__


# __air_pytorch_train_start__
import torch
import torch.nn as nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from ray import train
from ray.air import session
from ray.train.torch import TorchTrainer, to_air_checkpoint


def create_model(input_features):
    return nn.Sequential(
        nn.Linear(in_features=input_features, out_features=16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid(),
    )


def train_loop_per_worker(config):
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["num_epochs"]
    num_features = config["num_features"]

    # Get the Ray Dataset shard for this data parallel worker,
    # and convert it to a PyTorch Dataset.
    train_data = train.get_dataset_shard("train")

    def to_tensor_iterator(dataset, batch_size):
        data_iterator = dataset.iter_batches(
            batch_format="numpy", batch_size=batch_size
        )

        for d in data_iterator:
            yield torch.Tensor(d["input"]).float(), torch.Tensor(d["target"]).float()

    # Create model.
    model = create_model(num_features)
    model = train.torch.prepare_model(model)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for cur_epoch in range(epochs):
        for inputs, labels in to_tensor_iterator(train_data, batch_size):
            optimizer.zero_grad()
            predictions = model(inputs)
            train_loss = loss_fn(predictions, labels.unsqueeze(1))
            train_loss.backward()
            optimizer.step()
        loss = train_loss.item()
        session.report({"loss": loss}, checkpoint=to_air_checkpoint(model))


num_features = len(schema_order)

trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config={
        # Training batch size
        "batch_size": 128,
        # Number of epochs to train each task for.
        "num_epochs": 20,
        # Number of columns of datset
        "num_features": num_features,
        # Optimizer args.
        "lr": 0.001,
    },
    scaling_config={
        # Number of workers to use for data parallelism.
        "num_workers": 2,
        # Whether to use GPU acceleration.
        "use_gpu": False,
        # trainer_resources=0 so that the example works on Colab.
        "trainer_resources": {"CPU": 0},
    },
    datasets={"train": train_dataset},
    preprocessor=preprocessor,
)

result = trainer.fit()
print(f"Last result: {result.metrics}")
# Last result: {'loss': 0.6559339960416158, ...}
# __air_pytorch_train_end__

# __air_pytorch_tuner_start__
from ray import tune
from ray.tune.tuner import Tuner, TuneConfig
from ray.air.config import RunConfig

tuner = Tuner(
    trainer,
    param_space={"train_loop_config": {"lr": tune.uniform(0.0001, 0.01)}},
    tune_config=TuneConfig(num_samples=5, metric="loss", mode="min"),
)
result_grid = tuner.fit()
best_result = result_grid.get_best_result()
print("Best Result:", best_result)
# Best Result: Result(metrics={'loss': 0.278409322102863, ...})

checkpoint = best_result.checkpoint
# __air_pytorch_tuner_end__

# __air_pytorch_batchpred_start__
from ray.train.batch_predictor import BatchPredictor
from ray.train.torch import TorchPredictor

batch_predictor = BatchPredictor.from_checkpoint(
    checkpoint, TorchPredictor, model=create_model(num_features)
)

predicted_probabilities = batch_predictor.predict(test_dataset)
print("PREDICTED PROBABILITIES")
predicted_probabilities.show()
# {'predictions': array([1.], dtype=float32)}
# {'predictions': array([0.], dtype=float32)}
# __air_pytorch_batchpred_end__
