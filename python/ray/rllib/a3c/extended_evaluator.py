from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
import numpy as np
from ray.rllib.envs import create_and_wrap
from ray.rllib.optimizers import Evaluator
from ray.rllib.a3c.common import get_policy_cls
from ray.rllib.a3c.runner import A3CEvaluator
from ray.rllib.utils.filter import get_filter
from ray.rllib.utils.sampler import AsyncSampler
from ray.rllib.utils.process_rollout import process_rollout


class ShardA3CEvaluator(A3CEvaluator):

    def __init__(self, env_creator, config, logdir, pin_id=None):
        super(ShardA3CEvaluator, self).__init__(env_creator, config, logdir)

        if pin_id:
            try:
                import psutil
                p = psutil.Process()
                p.cpu_affinity([pin_id])
                print("Setting CPU Affinity to: ", pin_id)
            except Exception as e:
                print(e)
                pass

    def compute_deltas(self, *shards): # NEED object IDs
        old_weights = reconstruct_weights(shards)
        self.set_flat(old_weights)
        grad, _ = self.compute_gradients(self.sample())
        self.apply_gradients(grad)
        new_weights = self.get_flat()
        return shard(new_weights - old_weights, len(shards))

    def get_flat(self):
        return self.policy.variables.get_flat()

    def set_flat(self, weights):
        return self.policy.variables.set_flat(weights)


def setup_sharded(num_shards):
    ShardA3CEvaluator.compute_deltas = ray.method(
        num_return_vals=num_shards)(ShardA3CEvaluator.compute_deltas)

    return ray.remote(ShardA3CEvaluator)

def shard(array, num):
    return np.array_split(array, num)

def reconstruct_weights(shards):
    return np.concatenate(shards)

