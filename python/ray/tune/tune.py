from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time

from ray.tune.error import TuneError
from ray.tune.suggest import BasicVariantGenerator
from ray.tune.trial import Trial, DEBUG_PRINT_INTERVAL
from ray.tune.result import DEFAULT_RESULTS_DIR
from ray.tune.log_sync import wait_for_log_sync
from ray.tune.trial_runner import TrialRunner
from ray.tune.schedulers import (HyperBandScheduler, AsyncHyperBandScheduler,
                                 FIFOScheduler, MedianStoppingRule)
from ray.tune.web_server import TuneServer

logger = logging.getLogger(__name__)

_SCHEDULERS = {
    "FIFO": FIFOScheduler,
    "MedianStopping": MedianStoppingRule,
    "HyperBand": HyperBandScheduler,
    "AsyncHyperBand": AsyncHyperBandScheduler,
}


def _make_scheduler(args):
    if args.scheduler in _SCHEDULERS:
        return _SCHEDULERS[args.scheduler](**args.scheduler_config)
    else:
        raise TuneError("Unknown scheduler: {}, should be one of {}".format(
            args.scheduler, _SCHEDULERS.keys()))


def run_experiments(experiments=None,
                    search_alg=None,
                    scheduler=None,
                    restore_from_path=None,
                    checkpoint_dir=None,
                    checkpoint_freq=0,
                    with_server=False,
                    server_port=TuneServer.DEFAULT_PORT,
                    verbose=True,
                    queue_trials=False,
                    trial_executor=None,
                    raise_on_failed_trial=True):
    """Runs and blocks until all trials finish.

    Args:
        experiments (Experiment | list | dict): Experiments to run. Will be
            passed to `search_alg` via `add_configurations`.
        search_alg (SearchAlgorithm): Search Algorithm. Defaults to
            BasicVariantGenerator.
        scheduler (TrialScheduler): Scheduler for executing
            the experiment. Choose among FIFO (default), MedianStopping,
            AsyncHyperBand, and HyperBand.
        restore_from_path (str): Restores experiment execution state to
            given checkpoint path.
        checkpoint_dir (str): Path at which experiment checkpoints are stored.
            Defaults to DEFAULT_RESULTS_DIR.
        checkpoint_freq (int): How many trial results between
            checkpoints. A value of 0 (default) disables checkpointing.
        with_server (bool): Starts a background Tune server. Needed for
            using the Client API.
        server_port (int): Port number for launching TuneServer.
        verbose (bool): How much output should be printed for each trial.
        queue_trials (bool): Whether to queue trials when the cluster does
            not currently have enough resources to launch one. This should
            be set to True when running on an autoscaling cluster to enable
            automatic scale-up.
        trial_executor (TrialExecutor): Manage the execution of trials.
        raise_on_failed_trial (bool): Raise TuneError if there exists failed
            trial (of ERROR state) when the experiments complete.

    Examples:
        >>> experiment_spec = Experiment("experiment", my_func)
        >>> run_experiments(experiments=experiment_spec)

        >>> experiment_spec = {"experiment": {"run": my_func}}
        >>> run_experiments(experiments=experiment_spec)

        >>> run_experiments(
        >>>     experiments=experiment_spec,
        >>>     scheduler=MedianStoppingRule(...))

        >>> run_experiments(
        >>>     experiments=experiment_spec,
        >>>     search_alg=SearchAlgorithm(),
        >>>     scheduler=MedianStoppingRule(...))

    Returns:
        List of Trial objects, holding data for each executed trial.

    """

    if scheduler is None:
        scheduler = FIFOScheduler()

    if search_alg is None:
        search_alg = BasicVariantGenerator()

    runner = TrialRunner(
        search_alg,
        scheduler=scheduler,
        checkpoint_dir=checkpoint_dir or DEFAULT_RESULTS_DIR,
        checkpoint_freq=checkpoint_freq,
        launch_web_server=with_server,
        server_port=server_port,
        verbose=verbose,
        queue_trials=queue_trials,
        trial_executor=trial_executor)

    if restore_from_path:
        if not os.path.exists(restore_from_path):
            raise ValueError("Provided path invalid: %s" % restore_from_path)
        assert experiments is None, (
            "Simultaneous starting experiments and restoring not supported.")
        runner.restore(restore_from_path)
    else:
        search_alg.add_configurations(experiments)

    logger.info(runner.debug_string(max_debug=99999))
    last_debug = 0
    while not runner.is_finished():
        runner.step()
        if time.time() - last_debug > DEBUG_PRINT_INTERVAL:
            logger.info(runner.debug_string())
            last_debug = time.time()

    logger.info(runner.debug_string(max_debug=99999))

    wait_for_log_sync()

    errored_trials = []
    for trial in runner.get_trials():
        if trial.status != Trial.TERMINATED:
            errored_trials += [trial]

    if errored_trials:
        if raise_on_failed_trial:
            raise TuneError("Trials did not complete", errored_trials)
        else:
            logger.error("Trials did not complete: %s", errored_trials)

    return runner.get_trials()
