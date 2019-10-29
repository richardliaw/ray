from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from filelock import FileLock
import torch
import torch.nn as nn
import argparse
from ray import tune
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import logging

import ray
from ray.experimental.sgd.pytorch import (PyTorchTrainer, PyTorchTrainable)
from ray.experimental.sgd.pytorch.resnet import ResNet18

logger = logging.getLogger()

def preinit_hook(runner):
    print("NCCL DEBUG SET")
    # Need this for avoiding a connection restart issue
    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    os.environ["NCCL_LL_THRESHOLD"] = "0"
    os.environ["NCCL_DEBUG"] = "INFO"


def setup_model(model, runner, device=None):
    distribute = runner.world_size > 1
    if torch.cuda.is_available():
        if device is None:
            device = torch.device("cuda")
            model = model.to(device)
        # check to place model on cuda device
        if distribute:
            model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if distribute:
            model = torch.nn.parallel.DistributedDataParallelCPU(model)
    return model


def initializer(runner):
    config = runner.config
    runner.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])  # meanstd transformation

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    with FileLock(os.path.expanduser("~/data.lock")):
        trainset = torchvision.datasets.CIFAR10(
            root="~/data",
            train=True,
            download=True,
            transform=transform_train)
    train_sampler = (torch.utils.data.distributed.DistributedSampler(trainset))

    valset = torchvision.datasets.CIFAR10(
        root="~/data", train=False, download=False, transform=transform_test)
    val_sampler = (torch.utils.data.distributed.DistributedSampler(valset))

    runner.train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=32,
        num_workers=2,
        pin_memory=False,
        shuffle=(not train_sampler),
        sampler=train_sampler)

    runner.validation_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=32,
        num_workers=2,
        pin_memory=False,
        shuffle=(not val_sampler),
        sampler=val_sampler)

    model = ResNet18()
    logger.info("creating model!")
    runner.model = setup_model(model, runner)
    runner.optimizer = torch.optim.SGD(
        model.parameters(), lr=config.get("lr", 1e-2))

    runner.criterion = nn.CrossEntropyLoss().to(runner.device)


def train(runner):
    model = runner.model
    train_iterator = runner.train_loader
    criterion = runner.criterion
    optimizer = runner.optimizer

    model.train()
    train_loss, total_num, correct = 0, 0, 0
    for batch_idx, (data, target) in enumerate(train_iterator):
        # get small model update
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        train_loss += loss.item() * target.size(0)
        total_num += target.size(0)
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        optimizer.step()
        optimizer.zero_grad()
    stats = {
        "train_loss": train_loss / total_num,
        "train_acc": correct / total_num
    }
    runner.log_results(DONE=True, **stats)


def train_example(num_replicas=1, use_gpu=False):
    trainer1 = PyTorchTrainer(
        initializer=initializer,
        preinit_hook=preinit_hook,
        train_function=train,
        num_replicas=num_replicas,
        use_gpu=use_gpu,
        backend="nccl")
    stats = {}
    while "DONE" not in stats:
        stats = trainer1.step()
        print(stats)
    # print(trainer1.apply(validation_function))
    trainer1.shutdown()
    print("success!")


# def tune_example(num_replicas=1, use_gpu=False):
#     config = {
#         "model_creator": ResNet18,
#         "data_creator": cifar_creator,
#         "optimizer_creator": optimizer_creator,
#         "loss_creator": nn.CrossEntropyLoss,
#         "num_replicas": num_replicas,
#         "initialization_hook": initialization_hook,
#         "use_gpu": use_gpu,
#         "batch_size": 512,
#         "config": {
#             "lr": tune.choice([1e-4, 1e-3, 5e-3, 1e-2])
#         },
#         "backend": "nccl"
#     }

#     analysis = tune.run(
#         PyTorchTrainable,
#         num_samples=12,
#         config=config,
#         stop={"training_iteration": 2},
#         verbose=1)

#     return analysis.get_best_config(metric="validation_loss", mode="min")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ray-redis-address",
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
    parser.add_argument(
        "--tune", action="store_true", default=False, help="Tune training")

    args, _ = parser.parse_known_args()

    ray.init(address=args.ray_redis_address, log_to_driver=True)

    # if args.tune:
    #     tune_example(num_replicas=args.num_replicas, use_gpu=args.use_gpu)
    # else:
    train_example(num_replicas=args.num_replicas, use_gpu=args.use_gpu)
