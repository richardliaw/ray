import ray
import numpy as np
from ray.rllib.optimizers.optimizer import Optimizer
from ray.rllib.a3c.extended_evaluator import ShardA3CEvaluator, setup_sharded, shard

@ray.remote
class ParameterServer(object):
    def __init__(self, weight_shard: np.ndarray):
        # self.params = {k: v.copy() for k, v in weight_shard}
        self.params = weight_shard.copy()
        print(self.params.shape)

    def update_and_get_weights(self, deltas):
        print(sum(deltas), self.params.shape)
        self.params += deltas
        return self.get_weights()

    def get_weights(self):
        # weights = list(self.params.items())
        # weights =
        return self.params

    def ip(self):
        return ray.services.get_node_ip_address()

# def create_ps(weight_shard: np.ndarray):
#     RemotePS = ray.remote(ParameterServer)
#     return RemotePS.remote(weight_shard)


class ShardedPS():
    def __init__(self, weights, ps_count):
        self.ps_dict = {}

        # populate indices
        # name_sorted_keys = list(sorted(weight_dict))
        # size_sorted_weights = list(sorted(
        #     weight_dict, key=lambda k: -len(weight_dict[k].flatten())))
        # size = {i: 0 for i in range(ps_count)}
        # for k in size_sorted_weights:
        #     wsize = len(weight_dict[k].flatten())
        #     min_ps_idx = min(size, key=lambda j: size[j])
        #     size[min_ps_idx] += wsize
        #     self.indices[min_ps_idx].append(name_sorted_keys.index(k))

        # start pss
        for ps_id, weight_shard in enumerate(shard(weights, ps_count)):
            # keys = [name_sorted_keys[i] for i in self.get_indices(ps_id)]
            self.ps_dict[ps_id] = ParameterServer.remote(weight_shard)

    def update(self, sharded_deltas: list):
        weight_ids = []
        for ps_id, weight_shard in enumerate(sharded_deltas):
            weight_ids.append(
                self.ps_dict[ps_id].update_and_get_weights.remote(weight_shard))
        return weight_ids

    def get_weight_ids(self):
        return [self.ps_dict[ps_id].get_weights.remote() for ps_id in sorted(self.ps_dict)]


class PSOptimizer(Optimizer):
    def _init(self):
        weights = self.local_evaluator.get_flat()
        self.ps = ShardedPS(weights, self.config["ps_count"])

    def step(self):
        # send deltas to parameter servers
        # import ipdb; ipdb.set_trace()
        weight_ids = self.ps.get_weight_ids()
        worker_to_shard = {}
        shard_to_worker = {}
        for w in self.remote_evaluators:
            shards = w.compute_deltas.remote(*weight_ids)
            worker_to_shard[w] = shards
            shard_to_worker.update({t: w for t in shards})

        for i in range(10):
            [done_shard], _ = ray.wait(list(shard_to_worker))
            w = shard_to_worker[done_shard]

            all_w_shards = worker_to_shard.pop(w)
            [shard_to_worker.pop(s) for s in all_w_shards]
            weight_ids = self.ps.update(all_w_shards)

            shards = w.compute_deltas.remote(*weight_ids)
            worker_to_shard[w] = shards
            shard_to_worker.update({t: w for t in shards})
            print(shard_to_worker)


if __name__ == '__main__':
    from ray.rllib.a3c import DEFAULT_CONFIG
    import gym
    ray.init()
    env_creator = lambda: gym.make("CartPole-v0")
    config = DEFAULT_CONFIG.copy()
    config["use_lstm"] = False
    config["ps_count"] = 6
    config["num_workers"] = 2
    logdir = "/tmp/shard"

    local_evaluator = ShardA3CEvaluator(env_creator, config, logdir)
    # RemoteEAEvaluator = setup_sharded(len(local_evaluator.policy.get_weights()))
    RemoteEAEvaluator = setup_sharded(config["ps_count"])

    remotes = [RemoteEAEvaluator.remote(env_creator, config, logdir)
                    for i in range(config["num_workers"])]
    optimizer = PSOptimizer(config, local_evaluator, remotes)

    optimizer.step()
