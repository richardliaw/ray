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
        self.timers = {}
        self.timers["grad"] = TimerStat()

    def compute_flat_grad(self, *shards): # NEED object IDs
        """Fuses set_weights and compute_gradients for shards.
        Returns:
            delta_shards (list): list of shards
        """
        with self.timers["grad"]:
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

    def stats(self):

        stats = {k + "_ms": round(1000 * v.mean, 3) for k, v in self.timers.items()}
        self.timers["grad"] = TimerStat()
        return stats

    def loop(self, ps_dict, iterations=100):
        ps = PSClient(ps_dict)
        shard_ids = ps.get_weights()
        for i in range(iterations):
            shards = ray.get(shard_ids)
            deltas = self.compute_flad_grad(*shards)
            shard_ids = ps.update(deltas)




def setup_sharded(num_shards, force=False):
    ShardA3CEvaluator.compute_flat_grad = ray.method(
        num_return_vals=num_shards)(ShardA3CEvaluator.compute_flat_grad)
    if force:
        return ray.remote(num_gpus=1)(ShardA3CEvaluator)
    else:
        return ray.remote(ShardA3CEvaluator)

def reconstruct_weights(shards):
    return np.concatenate(shards)

