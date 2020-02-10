import os
import tempfile
from unittest.mock import patch

import pytest
import time
import torch
import torch.nn as nn
import torch.distributed as dist

import ray
from ray import tune
from ray.tests.conftest import ray_start_2_cpus  # noqa: F401
from ray.experimental.sgd.pytorch import PyTorchTrainer, PyTorchTrainable
from ray.experimental.sgd.pytorch.utils import (train, BATCH_COUNT, TEST_MODE,
                                                SCHEDULER_STEP)
from ray.experimental.sgd.utils import check_for_failure

from ray.experimental.sgd.pytorch.examples.train_example import (
    model_creator, optimizer_creator, data_creator, LinearDataset)


def test_test_mode(ray_start_2_cpus):  # noqa: F811
    trainer = PyTorchTrainer(
        model_creator,
        data_creator,
        optimizer_creator,
        loss_creator=lambda config: nn.MSELoss(),
        config={TEST_MODE: True},
        num_replicas=1)
    metrics = trainer.train()
    assert metrics[BATCH_COUNT] == 1

    val_metrics = trainer.validate()
    assert val_metrics[BATCH_COUNT] == 1


@pytest.mark.parametrize("num_replicas", [1, 2]
                         if dist.is_available() else [1])
def test_train(ray_start_2_cpus, num_replicas):  # noqa: F811
    trainer = PyTorchTrainer(
        model_creator,
        data_creator,
        optimizer_creator,
        loss_creator=lambda config: nn.MSELoss(),
        num_replicas=num_replicas)
    for i in range(3):
        train_loss1 = trainer.train()["train_loss"]
    validation_loss1 = trainer.validate()["validation_loss"]

    for i in range(3):
        train_loss2 = trainer.train()["train_loss"]
    validation_loss2 = trainer.validate()["validation_loss"]

    print(train_loss1, train_loss2)
    print(validation_loss1, validation_loss2)

    assert train_loss2 <= train_loss1
    assert validation_loss2 <= validation_loss1


@pytest.mark.parametrize("num_replicas", [1, 2]
                         if dist.is_available() else [1])
def test_multi_model(ray_start_2_cpus, num_replicas):  # noqa: F811
    def custom_train(config, models, dataloader, criterion, optimizers,
                     **kwargs):
        result = {}
        for i, (model, optimizer) in enumerate(zip(models, optimizers)):
            result["model_{}".format(i)] = train(config, model, dataloader,
                                                 criterion, optimizer)
        return result

    def multi_model_creator(config):
        return nn.Linear(1, 1), nn.Linear(1, 1)

    def multi_optimizer_creator(models, config):
        opts = [
            torch.optim.SGD(model.parameters(), lr=0.0001) for model in models
        ]
        return opts[0], opts[1]

    trainer1 = PyTorchTrainer(
        multi_model_creator,
        data_creator,
        multi_optimizer_creator,
        loss_creator=lambda config: nn.MSELoss(),
        train_function=custom_train,
        num_replicas=num_replicas)
    trainer1.train()

    filename = os.path.join(tempfile.mkdtemp(), "checkpoint")
    trainer1.save(filename)

    models1 = trainer1.get_model()

    trainer1.shutdown()

    trainer2 = PyTorchTrainer(
        multi_model_creator,
        data_creator,
        multi_optimizer_creator,
        loss_creator=lambda config: nn.MSELoss(),
        num_replicas=num_replicas)
    trainer2.restore(filename)

    os.remove(filename)

    models2 = trainer2.get_model()

    for model_1, model_2 in zip(models1, models2):

        model1_state_dict = model_1.state_dict()
        model2_state_dict = model_2.state_dict()

        assert set(model1_state_dict.keys()) == set(model2_state_dict.keys())

        for k in model1_state_dict:
            assert torch.equal(model1_state_dict[k], model2_state_dict[k])

    trainer2.shutdown()


@pytest.mark.parametrize("num_replicas", [1, 2]
                         if dist.is_available() else [1])
def test_multi_model_matrix(ray_start_2_cpus, num_replicas):  # noqa: F811
    def custom_train(config, model, dataloader, criterion, optimizer,
                     scheduler):
        if config.get("models", 1) > 1:
            assert len(model) == config["models"], config

        if config.get("optimizers", 1) > 1:
            assert len(optimizer) == config["optimizers"], config

        if config.get("schedulers", 1) > 1:
            assert len(scheduler) == config["schedulers"], config
        return {"done": 1}

    def multi_model_creator(config):
        models = []
        for i in range(config.get("models", 1)):
            models += [nn.Linear(1, 1)]
        return models[0] if len(models) == 1 else models

    def multi_optimizer_creator(models, config):
        optimizers = []
        main_model = models[0] if type(models) is list else models
        for i in range(config.get("optimizers", 1)):
            optimizers += [torch.optim.SGD(main_model.parameters(), lr=0.0001)]
        return optimizers[0] if len(optimizers) == 1 else optimizers

    def multi_scheduler_creator(optimizer, config):
        schedulers = []
        main_opt = optimizer[0] if type(optimizer) is list else optimizer
        for i in range(config.get("schedulers", 1)):
            schedulers += [
                torch.optim.lr_scheduler.StepLR(
                    main_opt, step_size=30, gamma=0.1)
            ]
        return schedulers[0] if len(schedulers) == 1 else schedulers

    for model_count in range(1, 3):
        for optimizer_count in range(1, 3):
            for scheduler_count in range(1, 3):
                trainer = PyTorchTrainer(
                    multi_model_creator,
                    data_creator,
                    multi_optimizer_creator,
                    loss_creator=nn.MSELoss,
                    scheduler_creator=multi_scheduler_creator,
                    train_function=custom_train,
                    num_replicas=num_replicas,
                    config={
                        "models": model_count,
                        "optimizers": optimizer_count,
                        "schedulers": scheduler_count
                    })
                trainer.train()
                trainer.shutdown()


@pytest.mark.parametrize("scheduler_freq", ["epoch", "batch"])
def test_scheduler_freq(ray_start_2_cpus, scheduler_freq):  # noqa: F811
    def custom_train(config, model, dataloader, criterion, optimizer,
                     scheduler):
        assert config[SCHEDULER_STEP] == scheduler_freq
        return {"done": 1}

    def scheduler_creator(optimizer, config):
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1)

    trainer = PyTorchTrainer(
        model_creator,
        data_creator,
        optimizer_creator,
        loss_creator=lambda config: nn.MSELoss(),
        scheduler_creator=scheduler_creator)

    for i in range(3):
        trainer.train()["train_loss"]
    trainer.shutdown()


@pytest.mark.parametrize("num_replicas", [1, 2]
                         if dist.is_available() else [1])
def test_tune_train(ray_start_2_cpus, num_replicas):  # noqa: F811

    config = {
        "model_creator": model_creator,
        "data_creator": data_creator,
        "optimizer_creator": optimizer_creator,
        "loss_creator": lambda config: nn.MSELoss(),
        "num_replicas": num_replicas,
        "use_gpu": False,
        "batch_size": 512,
        "backend": "gloo"
    }

    analysis = tune.run(
        PyTorchTrainable,
        num_samples=2,
        config=config,
        stop={"training_iteration": 2},
        verbose=1)

    # # checks loss decreasing for every trials
    for path, df in analysis.trial_dataframes.items():
        train_loss1 = df.loc[0, "train_loss"]
        train_loss2 = df.loc[1, "train_loss"]
        validation_loss1 = df.loc[0, "validation_loss"]
        validation_loss2 = df.loc[1, "validation_loss"]

        assert train_loss2 <= train_loss1
        assert validation_loss2 <= validation_loss1


@pytest.mark.parametrize("num_replicas", [1, 2]
                         if dist.is_available() else [1])
def test_save_and_restore(ray_start_2_cpus, num_replicas):  # noqa: F811
    trainer1 = PyTorchTrainer(
        model_creator,
        data_creator,
        optimizer_creator,
        loss_creator=lambda config: nn.MSELoss(),
        num_replicas=num_replicas)
    trainer1.train()

    filename = os.path.join(tempfile.mkdtemp(), "checkpoint")
    trainer1.save(filename)

    model1 = trainer1.get_model()

    trainer1.shutdown()

    trainer2 = PyTorchTrainer(
        model_creator,
        data_creator,
        optimizer_creator,
        loss_creator=lambda config: nn.MSELoss(),
        num_replicas=num_replicas)
    trainer2.restore(filename)

    os.remove(filename)

    model2 = trainer2.get_model()

    model1_state_dict = model1.state_dict()
    model2_state_dict = model2.state_dict()

    assert set(model1_state_dict.keys()) == set(model2_state_dict.keys())

    for k in model1_state_dict:
        assert torch.equal(model1_state_dict[k], model2_state_dict[k])


def test_fail_with_recover(ray_start_2_cpus):  # noqa: F811
    if not dist.is_available():
        return

    def single_loader(config):
        return LinearDataset(2, 5, size=1000000)

    def step_with_fail(self):
        worker_stats = [w.step.remote() for w in self.workers]
        if self._num_failures < 3:
            time.sleep(1)  # Make the batch will fail correctly.
            self.workers[0].__ray_kill__()
        success = check_for_failure(worker_stats)
        return success, worker_stats

    with patch.object(PyTorchTrainer, "_train_step", step_with_fail):
        trainer1 = PyTorchTrainer(
            model_creator,
            single_loader,
            optimizer_creator,
            batch_size=100000,
            loss_creator=lambda config: nn.MSELoss(),
            num_replicas=2)

        with pytest.raises(RuntimeError):
            trainer1.train(max_retries=1)


def test_resize(ray_start_2_cpus):  # noqa: F811
    if not dist.is_available():
        return

    def single_loader(config):
        return LinearDataset(2, 5, size=1000000)

    def step_with_fail(self):
        worker_stats = [w.step.remote() for w in self.workers]
        if self._num_failures < 1:
            time.sleep(1)  # Make the batch will fail correctly.
            self.workers[0].__ray_kill__()
        success = check_for_failure(worker_stats)
        return success, worker_stats

    with patch.object(PyTorchTrainer, "_train_step", step_with_fail):
        trainer1 = PyTorchTrainer(
            model_creator,
            single_loader,
            optimizer_creator,
            batch_size=100000,
            loss_creator=lambda config: nn.MSELoss(),
            num_replicas=2)

        @ray.remote
        def try_test():
            import time
            time.sleep(100)

        try_test.remote()
        trainer1.train(max_retries=1)
        assert len(trainer1.workers) == 1


def test_fail_twice(ray_start_2_cpus):  # noqa: F811
    if not dist.is_available():
        return

    def single_loader(config):
        return LinearDataset(2, 5, size=1000000)

    def step_with_fail(self):
        worker_stats = [w.step.remote() for w in self.workers]
        if self._num_failures < 2:
            time.sleep(1)
            self.workers[0].__ray_kill__()
        success = check_for_failure(worker_stats)
        return success, worker_stats

    with patch.object(PyTorchTrainer, "_train_step", step_with_fail):
        trainer1 = PyTorchTrainer(
            model_creator,
            single_loader,
            optimizer_creator,
            batch_size=100000,
            loss_creator=lambda config: nn.MSELoss(),
            num_replicas=2)

        trainer1.train(max_retries=2)
