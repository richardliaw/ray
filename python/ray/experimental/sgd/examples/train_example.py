from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import argparse
from ray import tune
from ray.experimental.sgd.pytorch.pytorch_trainer import (PyTorchTrainer,
                                                          PyTorchTrainable)

import ray
from ray.experimental.sgd.pytorch import PyTorchTrainer
from ray.experimental.sgd.tests.pytorch_utils import (
    resnet_creator, xe_optimizer_creator, cifar_creator)


def initialization_hook(runner):
    print("NCCL DEBUG SET")
    # Need this for avoiding a connection restart issue
    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    os.environ["NCCL_LL_THRESHOLD"] = "0"
    os.environ["NCCL_DEBUG"] = "INFO"


def train(model, train_iterator, criterion, optimizer):
    model.train()
    train_loss, total_num, correct = 0, 0, 0
    for batch_idx, (data, target) in enumerate(train_iterator):
        # get small model update
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)  # / float(large_ratio)
        loss.backward()
        train_loss += loss.item() * target.size(0)  # * float(large_ratio)
        total_num += target.size(0)
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        optimizer.step()
        optimizer.zero_grad()
    stats = {"train_loss": train_loss / total_num}
    return stats


def train_example(num_replicas=1, use_gpu=False):
    trainer1 = PyTorchTrainer(
        resnet_creator,
        cifar_creator,
        xe_optimizer_creator,
        train_function=train,
        num_replicas=num_replicas,
        use_gpu=use_gpu,
        batch_size=2048,
        backend="nccl")
    stats = trainer1.train()
    print(stats)
    trainer1.train()
    trainer1.shutdown()
    print("success!")


def tune_example(num_replicas=1, use_gpu=False):
    config =

    # analysis = tune.run(PyTorchTrainable, num_samples=1, config=config)
    analysis = tune.run(
        PyTorchTrainer.make_trainable(
            resnet_creator,
            cifar_creator,
            xe_optimizer_creator,
            nn.MSELoss,
            train_function=train_function,
        ),
        num_samples=12,
        config={"num_replicas": num_replicas, "use_gpu": use_gpu},
        stop={"training_iteration": 10},
        verbose=1)
    return analysis.get_best_config(metric="validation_loss", mode="min")


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

    ray.init(redis_address=args.redis_address)
    tune_example(num_replicas=args.num_replicas, use_gpu=args.use_gpu)
