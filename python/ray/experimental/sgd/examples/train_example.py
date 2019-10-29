"""
This file holds code for a Training guide for PytorchSGD in the documentation.

It ignores yapf because yapf doesn't allow comments right after code blocks,
but we put comments right after code blocks to prevent large white spaces
in the documentation.
"""

# yapf: disable
# __torch_train_example__
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

import time
import torch
import torch.nn as nn

from ray.experimental.sgd import utils

from ray.experimental.sgd.pytorch.pytorch_trainer import PyTorchTrainer


class LinearDataset(torch.utils.data.Dataset):
    """y = a * x + b"""

    def __init__(self, a, b, size=1000):
        x = np.random.random(size).astype(np.float32) * 10
        x = np.arange(0, 10, 10 / size, dtype=np.float32)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(a * x + b)

    def __getitem__(self, index):
        return self.x[index, None], self.y[index, None]

    def __len__(self):
        return len(self.x)


def initializer(runner):
    training_set = LinearDataset(2, 5)
    validation_set = LinearDataset(2, 5, size=400)
    runner.model = nn.Linear(1, 1).to(runner.device)
    runner.criterion = nn.MSELoss().to(runner.device)
    runner.optimizer = torch.optim.SGD(runner.model.parameters(), lr=1e-4)

    # logger.debug("Creating dataset")
    runner.train_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=False)

    runner.validation_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=False)


def train_function(runner):
    """Runs 1 training epoch"""
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()

    timers = {k: utils.TimerStat() for k in ["d2h", "fwd", "grad", "apply"]}

    # switch to train mode
    runner.model.train()
    end = time.time()

    for i, (features, target) in enumerate(runner.train_loader):
        # Create non_blocking tensors for distributed training
        with timers["d2h"]:
            if torch.cuda.is_available():
                features = features.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

        # compute output
        with timers["fwd"]:
            output = runner.model(features)
            loss = runner.criterion(output, target)
            # measure accuracy and record loss
            losses.update(loss.item(), features.size(0))

        with timers["grad"]:
            # compute gradients in a backward pass
            runner.optimizer.zero_grad()
            loss.backward()

        with timers["apply"]:
            # Call step of optimizer to update model params
            runner.optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    stats = {
        "batch_time": batch_time.avg,
        "batch_processed": losses.count,
        "train_loss": losses.avg,
        "data_time": data_time.avg,
    }
    stats.update({k: t.mean for k, t in timers.items()})
    runner.log_results(DONE=True, **stats)


def validation_function(runner):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()

    # switch to evaluate mode
    runner.model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (features, target) in enumerate(runner.validation_loader):

            if torch.cuda.is_available():
                features = features.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output = runner.model(features)
            loss = runner.criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.item(), features.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return {"batch_time": batch_time.avg, "validation_loss": losses.avg}


def train_example(num_replicas=1, use_gpu=False):
    trainer1 = PyTorchTrainer(
        initializer=initializer,
        train_function=train_function,
        num_replicas=num_replicas,
        use_gpu=use_gpu,
        backend="gloo")
    for i in range(4):
        print(trainer1.step())
    print(trainer1.apply(validation_function))
    trainer1.shutdown()
    print("success!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--redis-address",
        required=False,
        type=str,
        help="the address to use for Redis")
    parser.add_argument(
        "--num-replicas",
        "-n",
        type=int,
        default=1,
        help="Sets number of replicas for training.")
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=False,
        help="Enables GPU training")
    args, _ = parser.parse_known_args()

    import ray

    ray.init(redis_address=args.redis_address)
    train_example(num_replicas=args.num_replicas, use_gpu=args.use_gpu)
