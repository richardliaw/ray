from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import ray

from ray.rllib.hpsearch.experiment import (
    ExperimentState, PENDING, RUNNING, TERMINATED)
from ray.rllib.hpsearch.utils import gpu_count

class ExperimentRunner(object):

    def __init__(self, experiments):
        self._experiments = experiments
        self._status = {e: ExperimentState(e) for e in self._experiments}
        self._pending = {}
        # TODO(ekl) query the ray cluster for these counts
        self._avail_resources = {
            'cpu': multiprocessing.cpu_count(),
            'gpu': gpu_count(),
        }
        self._committed_resources = {k: 0 for k in self._avail_resources}

    def is_finished(self):
        for (exp, status) in self._status.items():
            if status.state in [PENDING, RUNNING]:
                return False
        return True

    def can_launch_more(self):
        exp = self._get_runnable()
        return exp is not None

    def launch_experiment(self):
        exp = self._get_runnable()
        self._status[exp].state = RUNNING
        self._commit_resources(exp.resource_requirements())
        exp.start()
        self._pending[exp.train_remote()] = exp

    def process_events(self):
        [result_id], _ = ray.wait(self._pending.keys())
        exp = self._pending[result_id]
        del self._pending[result_id]
        result = ray.get(result_id)
        print("result", result)
        status = self._status[exp]
        status.last_result = result

        if exp.should_stop(result):
            status.state = TERMINATED
            self._return_resources(exp.resource_requirements())
            exp.stop()
        else:
            # TODO(rliaw): This implements checkpoint in a blocking manner
            if status.should_checkpoint():
                status.set_cp_path(exp.checkpoint())
            self._pending[exp.train_remote()] = exp

        # TODO(ekl) also switch to other experiments if the current one
        # doesn't look promising, i.e. bandits


    def _get_runnable(self):
        for exp in self._experiments:
            status = self._status[exp]
            if (status.state == PENDING and
                    self._has_resources(exp.resource_requirements())):
                return exp
        return None

    def _has_resources(self, resources):
        for k, v in resources.items():
            if self._avail_resources[k] < v:
                return False
        return True

    def _commit_resources(self, resources):
        for k, v in resources.items():
            self._avail_resources[k] -= v
            self._committed_resources[k] += v
            assert self._avail_resources[k] >= 0

    def _return_resources(self, resources):
        for k, v in resources.items():
            self._avail_resources[k] += v
            self._committed_resources[k] -= v
            assert self._committed_resources[k] >= 0

    def debug_string(self):
        statuses = [
            ' - {}:\t{}'.format(e, self._status[e]) for e in self._experiments]
        return 'Available resources: {}'.format(self._avail_resources) + \
            '\nCommitted resources: {}'.format(self._committed_resources) + \
            '\nAll experiments:\n' + '\n'.join(statuses)

