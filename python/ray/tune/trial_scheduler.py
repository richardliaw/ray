from __future__ import absolute_import
from __future__ import division

from ray.tune.trial import Trial


class TrialScheduler(object):
    CONTINUE = "CONTINUE"
    PAUSE = "PAUSE"
    STOP = "STOP"

    def on_trial_result(self, trial_runner, trial, result):
        """Called on each intermediate result returned by a trial.

        At this point, the trial scheduler can make a decision by returning
        one of CONTINUE, PAUSE, and STOP."""

        raise NotImplementedError

    def choose_trial_to_run(self, trial_runner, trials):
        """Called to choose a new trial to run.

        This should return one of the trials in trial_runner that is in
        the PENDING or PAUSED state."""

        raise NotImplementedError

    def debug_string(self):
        """Returns a human readable message for printing to the console."""

        raise NotImplementedError


class FIFOScheduler(TrialScheduler):
    def on_trial_result(self, trial_runner, trial, result):
        return TrialScheduler.CONTINUE

    def choose_trial_to_run(self, trial_runner):
        for trial in trial_runner.get_trials():
            if (trial.status == Trial.PENDING and
                    trial_runner.has_resources(trial.resources)):
                return trial
        return None

    def debug_string(self):
        return "Using FIFO scheduling algorithm."
