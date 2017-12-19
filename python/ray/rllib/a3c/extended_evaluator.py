from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray.rllib.envs import create_and_wrap
from ray.rllib.optimizers import Evaluator
from ray.rllib.a3c.common import get_policy_cls
from ray.rllib.a3c.runner import A3CEvaluator
from ray.rllib.utils.filter import get_filter
from ray.rllib.utils.sampler import AsyncSampler
from ray.rllib.utils.process_rollout import process_rollout


class ShardA3CEvaluator(A3CEvaluator):

    def _init(self):
        super(ExtendedA3CEvaluator, self)._init()
        self.ps_count = self.config["ps_count"]

    def compute_deltas(self, *shards):
        weight_list = []
        for s in shards:
            weight_list += s

        weights = {k: delt for k, delt in weight_list}
        self.policy.set_weights(weights)
        grad, _ = self.compute_gradients(self.sample())
        self.apply_gradients(grad)
        new_weights = self.get_weights()
        return sorted([(k, new_weights[k] - weights[k]) for k in new_weights]) # this is split


def setup_sharded(rets):
    ShardA3CEvaluator.compute_deltas = ray.method(
        num_return_vals=rets)(ShardA3CEvaluator.compute_deltas)

    return ray.remote(ShardA3CEvaluator)
