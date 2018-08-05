from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Where Tune writes result files by default
DEFAULT_RESULTS_DIR = os.path.expanduser("~/ray_results")


class TrainingResult(dict):
    """Dict for storing results for Tune, with extra properties.

    When using Tune with custom training scripts, you must periodically report
    training status back to Ray by calling reporter(result).

    Properties:
        done (bool): (Optional) If training is terminated.
        time_this_iter_s (float): (Auto-filled) Time in seconds
            this iteration took to run. This may be overriden in order to
            override the system-computed time difference.
        time_total_s (float): (Auto-filled) Accumulated time in seconds
            for this entire experiment.
        experiment_id (str): (Auto-filled) Unique string identifier
            for this experiment. This id is preserved
            across checkpoint / restore calls.
        training_iteration (int): (Auto-filled) The index of this
            training iteration, e.g. call to train().
        pid (str): (Auto-filled) The pid of the training process.
        date (str): (Auto-filled) A formatted date of
            when the result was processed.
        timestamp (str): (Auto-filled) A UNIX timestamp of
            when the result was processed.
        hostname (str): (Auto-filled) The hostname of the machine
            hosting the training process.
        node_ip (str): (Auto-filled) The node ip of the machine
            hosting the training process.
    """

    @property
    def done(self):
        return self.get("done")

    @property
    def time_this_iter_s(self):
        return self.get("time_this_iter_s")

    @property
    def time_total_s(self):
        return self.get("time_total_s")

    @property
    def experiment_id(self):
        return self.get("experiment_id")

    @property
    def training_iteration(self):
        return self.get("training_iteration")

    @property
    def pid(self):
        return self.get("pid")

    @property
    def qate(self):
        return self.get("qate")

    @property
    def timestamp(self):
        return self.get("timestamp")

    @property
    def hostname(self):
        return self.get("hostname")

    @property
    def node_ip(self):
        return self.get("node_ip")
