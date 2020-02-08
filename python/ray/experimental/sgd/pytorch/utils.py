import collections
import time
import torch

from ray.experimental.sgd.utils import TimerStat

amp = None

try:
    from apex import amp
except ImportError:
    pass

USE_FP16 = "use_fp16"

SCHEDULER_STEP = "scheduler_step"
SCHEDULER_STEP_BATCH = "batch"
SCHEDULER_STEP_EPOCH = "epoch"

VALID_SCHEDULER_STEP = set([SCHEDULER_STEP_BATCH, SCHEDULER_STEP_EPOCH])


def train(config, model, train_iterator, criterion, optimizer, scheduler=None):
    """Runs one standard training pass over the train_iterator.

    This function automatically measures timing for various operations such
    as host to device transfer, gradient calculation, and gradient application.

    It also automatically detects and places the data on the given GPU device
    if available.

    The scheduler will only be called at a batch or epoch frequency, depending
    on the user parameter. If using a scheduler that depends on validation
    loss, you must provide a custom training function.

    Raises:
        ValueError if multiple models/optimizers/schedulers are provided. You
            are expected to have a custom training function if you wish
            to use multiple models/optimizers/schedulers.

    Args:
        config: (dict): A user configuration provided into the Trainer
            constructor.
        model: The model as created by the model_creator.
        train_iterator: An iterator created from the DataLoader which
            wraps the provided Dataset.
        criterion: The loss object created by the loss_creator.
        optimizer: The torch.optim.Optimizer object
            as created by the optimizer_creator
        scheduler (optional): The torch.optim.lr_scheduler object
            as created by the scheduler_creator.

    Returns:
        A dict of metrics from training.
    """
    if isinstance(model, collections.Iterable) or isinstance(
            optimizer, collections.Iterable) or isinstance(
                scheduler, collections.Iterable):
        raise ValueError(
            "Need to provide custom training function if using multi-model "
            "or multi-scheduler or multi-optimizer training.")

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    timers = {k: TimerStat() for k in ["h2d", "fwd", "grad", "apply"]}

    # switch to train mode
    model.train()

    end = time.time()

    for i, (features, target) in enumerate(train_iterator):
        # measure data loading time
        data_time.update(time.time() - end)

        # Create non_blocking tensors for distributed training
        with timers["h2d"]:
            if torch.cuda.is_available():
                features = features.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

        # compute output
        with timers["fwd"]:
            output = model(features)
            loss = criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.item(), features.size(0))

        with timers["grad"]:
            # compute gradients in a backward pass
            optimizer.zero_grad()

            if config.get(USE_FP16):
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

        with timers["apply"]:
            # Call step of optimizer to update model params
            optimizer.step()

        if scheduler and config.get(SCHEDULER_STEP) == SCHEDULER_STEP_BATCH:
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if config.get(SCHEDULER_STEP) == SCHEDULER_STEP_EPOCH:
        scheduler.step()

    stats = {
        "batch_time": batch_time.avg,
        "batch_processed": losses.count,
        "train_loss": losses.avg,
        "data_time": data_time.avg,
    }
    stats.update({k: t.mean for k, t in timers.items()})
    return stats


def validate(config, model, val_iterator, criterion, scheduler):
    """Runs one standard validation pass over the val_iterator.

    This function automatically measures timing for various operations such
    as host to device transfer and processing time for the batch.

    It also automatically detects and places the data on the given GPU device
    if available.

    Raises:
        ValueError if multiple models/schedulers are provided. You
            are expected to have a custom training function if you wish
            to use multiple models/schedulers.

    Args:
        config: (dict): A user configuration provided into the Trainer
            constructor.
        model: The model as created by the model_creator.
        train_iterator: An iterator created from the DataLoader which
            wraps the provided Dataset.
        criterion: The loss object created by the loss_creator.
        scheduler (optional): The torch.optim.lr_scheduler object
            as created by the scheduler_creator.

    Returns:
        A dict of metrics from the evaluation.
    """

    if isinstance(model, collections.Iterable) or isinstance(
            scheduler, collections.Iterable):
        raise ValueError(
            "Need to provide custom validation function if using multi-model "
            "or multi-scheduler training.")
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        end = time.time()
        for i, (features, target) in enumerate(val_iterator):

            if torch.cuda.is_available():
                features = features.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(features)
            loss = criterion(output, target)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # measure accuracy and record loss
            losses.update(loss.item(), features.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    stats = {"batch_time": batch_time.avg, "validation_loss": losses.avg}
    stats.update(mean_accuracy=correct / total)
    return stats


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
