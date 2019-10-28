from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.utils.data
import logging
import time
import inspect
import threading
import traceback
from six.moves import queue

import ray
from ray.experimental.sgd.pytorch import pytorch_utils
from ray.experimental.sgd import utils

logger = logging.getLogger(__name__)

TIME_THIS_ITER_S = "time_this_iter_s"


class PyTorchRunner(object):
    """Manages a PyTorch model for training."""

    def __init__(self, train_function, validation_function, config=None):
        """Initializes the runner."""

        self.config = config or {}

        self.epoch = 0
        self._timers = {
            k: utils.TimerStat(window_size=1)
            for k in [
                "setup_proc", "setup_model", "get_state", "set_state",
                "validation", "training"
            ]
        }

        self._train_function = train_function
        self._validation_function = validation_function

        # the runner thread is not started until the first call to _train
        self._train_runner = None
        self._validation_runner = None

    def get_node_ip(self):
        """Returns the IP address of the current node."""
        return ray.services.get_node_ip_address()

    def find_free_port(self):
        """Finds a free port on the current node."""
        return utils.find_free_port()

    def stats(self):
        """Returns a dictionary of statistics collected."""
        stats = {"epoch": self.epoch}
        for k, t in self._timers.items():
            stats[k + "_time_mean"] = t.mean
            stats[k + "_time_total"] = t.sum
            t.reset()
        return stats

    def train_step(self):
        if self._train_runner is None:
            self._mode = "TRAIN"
            self._train_runner = _RunnerThread(
                lambda: self._train_function(self), self._error_queue)

        assert self._validation_mode is False
        result = self._train_runner.step()

        if "DONE" in result:
            self._train_runner.stop()
            self._train_runner = None
            self._mode = None
        return result

    def validation_step(self):
        if self._validation_runner is None:
            self._mode = "VALID"
            self._validation_runner = _RunnerThread(
                lambda: self._validation_function(self), self._error_queue)

        assert self._mode == "VALID"
        result = self._validation_runner.step()

        if "DONE" in result:
            self._validation_runner.stop()
            self._validation_runner = None
            self._mode = None
        return result

    def checkpoint(self):
        raise NotImplementedError

    def restore(self, data):
        raise NotImplementedError

    def shutdown(self):
        raise NotImplementedError

    def apply(self, fn):
        return fn(self)

    @property
    def logdir(self):
        return self._logdir

    def log_result(self, **results):
        """This blocks by default"""
        self._thread_manager.report(**results)


# Time between FunctionRunner checks when fetching
# new results after signaling the reporter to continue
RESULT_FETCH_TIMEOUT = 0.2

ERROR_REPORT_TIMEOUT = 10
ERROR_FETCH_TIMEOUT = 1


class ThreadManager():
    def __init__(self, entrypoint):
        # Queue for passing errors back from the thread runner. The error queue
        # has a max size of one to prevent stacking error and force error
        # reporting to block until finished.
        self._error_queue = queue.Queue(1)

        # Semaphore for notifying the reporter to continue with the computation
        # and to generate the next result.
        self._continue_semaphore = threading.Semaphore(0)

        # Queue for passing results between threads
        self._results_queue = queue.Queue(1)
        self._last_result = {}

    def report(self, **results):
        assert not isinstance(threading.current_thread(),
                              threading._MainThread)
        assert self._last_report_time is not None, (
            "StatusReporter._start() must be called before the first "
            "report __call__ is made to ensure correct runtime metrics.")

        # time per iteration is recorded directly in the reporter to ensure
        # any delays in logging results aren't counted
        report_time = time.time()
        if TIME_THIS_ITER_S not in results:
            results[TIME_THIS_ITER_S] = report_time - self._last_report_time
        self._last_report_time = report_time

        # add results to a thread-safe queue
        self._results_queue.put(results.copy(), block=True)

        # This blocks until notification from the FunctionRunner that the last
        # result has been returned to Tune and that the function is safe to
        # resume training.
        self._continue_semaphore.acquire()

    def step(self):
        """Lets the running thread run until a logging statement.

        If the RunnerThread finishes without reporting "done",
        Tune will automatically provide a magic keyword __duplicate__
        along with a result with "done=True". The TrialRunner will handle the
        result accordingly (see tune/trial_runner.py).
        """
        if self._runner.is_alive():
            # if started and alive, inform the reporter to continue and
            # generate the next result
            self._continue_semaphore.release()
        else:
            # if not alive, try to start
            self._last_report_time = time.time()
            try:
                self._runner.start()
            except RuntimeError:
                # If this is reached, it means the thread was started and is
                # now done or has raised an exception.
                pass

        result = None
        while result is None and self._runner.is_alive():
            # fetch the next produced result
            try:
                result = self._results_queue.get(
                    block=True, timeout=RESULT_FETCH_TIMEOUT)
            except queue.Empty:
                pass

        # if no result were found, then the runner must no longer be alive
        if result is None:
            # Try one last time to fetch results in case results were reported
            # in between the time of the last check and the termination of the
            # thread runner.
            try:
                result = self._results_queue.get(block=False)
            except queue.Empty:
                pass

        # check if error occured inside the thread runner
        if result is None:
            # only raise an error from the runner if all results are consumed
            self.report_thread_runner_error(block=True)

            # Under normal conditions, this code should never be reached since
            # this branch should only be visited if the runner thread raised
            # an exception. If no exception were raised, it means that the
            # runner thread never reported any results which should not be
            # possible when wrapping functions with `wrap_function`.
            raise Exception(
                ("Wrapped function ran until completion without reporting "
                 "results or raising an exception."))

        else:
            if not self._error_queue.empty():
                logger.warning(
                    ("Runner error waiting to be raised in main thread. "
                     "Logging all available results first."))

        # This keyword appears if the train_func using the Function API
        # finishes without "done=True". This duplicates the last result, but
        # the TrialRunner will not log this result again.
        if "__duplicate__" in result:
            new_result = self._last_result.copy()
            new_result.update(result)
            result = new_result

        self._last_result = result
        return result

    def stop(self):
        # If everything stayed in sync properly, this should never happen.
        if not self._results_queue.empty():
            logger.warning(
                ("Some results were added after the trial stop condition. "
                 "These results won't be logged."))

        # Check for any errors that might have been missed.
        self.report_thread_runner_error()

    def report_thread_runner_error(self, block=False):
        try:
            err_tb_str = self._error_queue.get(
                block=block, timeout=ERROR_FETCH_TIMEOUT)
            raise Exception(("Trial raised an exception. Traceback:\n{}"
                             .format(err_tb_str)))
        except queue.Empty:
            pass


class _RunnerThread(threading.Thread):
    """Supervisor thread that runs your script."""

    def __init__(self, entrypoint, error_queue):
        threading.Thread.__init__(self)
        self._entrypoint = entrypoint
        self._error_queue = error_queue
        self.daemon = True

    def run(self):
        try:
            self._entrypoint()
        except StopIteration:
            logger.debug(
                ("Thread runner raised StopIteration. Interpreting it as a "
                 "signal to terminate the thread without error."))
        except Exception as e:
            logger.exception("Runner Thread raised error.")
            try:
                # report the error but avoid indefinite blocking which would
                # prevent the exception from being propagated in the unlikely
                # case that something went terribly wrong
                err_tb_str = traceback.format_exc()
                self._error_queue.put(
                    err_tb_str, block=True, timeout=ERROR_REPORT_TIMEOUT)
            except queue.Full:
                logger.critical(
                    ("Runner Thread was unable to report error to main "
                     "function runner thread. This means a previous error "
                     "was not processed. This should never happen."))
            raise e
