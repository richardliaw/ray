from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
import numpy as np
from ray.rllib.optimizers import Evaluator
from ray.rllib.a3c.common import get_policy_cls
from ray.rllib.a3c.a3c_evaluator import A3CEvaluator
from ray.rllib.utils.filter import get_filter
from ray.rllib.utils.process_rollout import process_rollout
from ray.rllib.utils.timer import TimerStat


class ShardA3CEvaluator(A3CEvaluator):

    def __init__(self, registry, env_creator, config, logdir, start_sampler=True):
        super(ShardA3CEvaluator, self).__init__(
            registry, env_creator, config, logdir, start_sampler)

        shape = lambda tensor: tuple(tensor.get_shape().as_list())
        assert all(shape(v) == shape(g) for v, g in zip(
            self.policy.variables.variables.values(), self.policy.grads))
        self.grad_timer = TimerStat()

    def compute_flat_grad(self, *shards): # NEED object IDs
        """Fuses set_weights and compute_gradients for shards.
        Returns:
            delta_shards (list): list of shards
        """
        with self.grad_timer:
            old_weights = reconstruct_weights(shards)
            self.set_flat(old_weights)
            grads = super(ShardA3CEvaluator, self).compute_gradients(self.sample())
            flattened = np.concatenate([g.flatten() for g in grads])
            return shard(flattened, len(shards))

    def pin(self, cpu_id):
        try:
            import psutil
            p = psutil.Process()
            p.cpu_affinity([cpu_id])
            print("Setting CPU Affinity to: ", cpu_id)
        except Exception as e:
            print(e)

    def get_flat(self):
        return self.policy.variables.get_flat()

    def set_flat(self, weights):
        return self.policy.variables.set_flat(weights)

    # def stats(self):
    #     stats =  {
    #         "setup_time_ms": round(1000 * self.grad_timer.mean, 3)}

    #     self.grad_timer = TimerStat()
    #     return stats



def setup_sharded(num_shards, force=False):
    ShardA3CEvaluator.compute_flat_grad = ray.method(
        num_return_vals=num_shards)(ShardA3CEvaluator.compute_flat_grad)
    if force:
        return ray.remote(num_gpus=1)(ShardA3CEvaluator)
    else:
        return ray.remote(ShardA3CEvaluator)

def shard(array, num):
    rets = np.array_split(array, num)
    return rets

def reconstruct_weights(shards):
    return np.concatenate(shards)

