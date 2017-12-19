import ray
import numpy as np
from ray.rllib.optimizers.optimizer import Optimizer

class ParameterServer(object):
    def __init__(self, weight_shard: list):
        self.params = {k: v.copy() for k, v in weight_shard}

    def update_and_get_weights(self, *deltas):
        print([k for k, v in deltas])
        for k, delt in deltas:
            self.params[k] += delt
        return self.get_weights()

    def get_weights(self):
        weights = list(self.params.items())
        print("Loading", list(zip(*weights))[0])
        return weights

    def ip(self):
        return ray.services.get_node_ip_address()

def create_ps(weight_shard: list):
    count = len(weight_shard)
    ParameterServer.update_and_get_weights = ray.method(
         num_return_vals=count)(ParameterServer.update_and_get_weights)
    ParameterServer.get_weights = ray.method(
         num_return_vals=count)(ParameterServer.get_weights)
    RemotePS = ray.remote(ParameterServer)
    return RemotePS.remote(weight_shard)


class ShardedPS():
    def __init__(self, weight_dict, ps_count):
        self.ps_dict = {}
        self._num_weights = len(weight_dict)
        self.indices = {i: [] for i in range(ps_count)}

        # populate indices
        name_sorted_keys = list(sorted(weight_dict))
        size_sorted_weights = list(sorted(
            weight_dict, key=lambda k: -len(weight_dict[k].flatten())))
        size = {i: 0 for i in range(ps_count)}
        for k in size_sorted_weights:
            wsize = len(weight_dict[k].flatten())
            min_ps_idx = min(size, key=lambda j: size[j])
            size[min_ps_idx] += wsize
            self.indices[min_ps_idx].append(name_sorted_keys.index(k))

        # start pss
        for ps_id in range(ps_count):
            keys = [name_sorted_keys[i] for i in self.get_indices(ps_id)]
            self.ps_dict[ps_id] = create_ps(
                [(k, weight_dict[k].copy()) for k in keys])


    def get_indices(self, ps_id):
        return self.indices[ps_id]

    def update(self, sorted_deltas: list):
        assert len(sorted_deltas) == self._num_weights
        new_weights = []
        for ps_id, ps in self.ps_dict.items():
            shard = [sorted_deltas[i] for i in self.get_indices(ps_id)]
            new_weights.extend(
                ps.update_and_get_weights.remote(*shard))
        return new_weights

    def get_weight_ids(self):
        weight_ids = []
        for ps in self.ps_dict.values():
            weight_ids.extend(ps.get_weights.remote())
        return weight_ids


class PSOptimizer(Optimizer):
    def _init(self):
        weights = self.local_evaluator.policy.get_weights()
        self.ps = ShardedPS(weights, self.config["ps_count"])

    def step(self):
        # send deltas to parameter servers
        # import ipdb; ipdb.set_trace()
        weight_ids = self.ps.get_weight_ids()

        tasks = []
        for w in self.remote_evaluators:
            tasks.append((w, self.update_ps(w, weight_ids)))

        for i in range(10):
            w, next_weight_list = tasks.pop(0)
            ray.wait(next_weight_list) # is this needed?
            tasks.append((w, self.update_ps(w, next_weight_list)))

    def update_ps(self, worker, cur_weights):
        delta_ids = worker.compute_deltas.remote(*cur_weights)
        weight_ids = self.ps.update(delta_ids)
        return weight_ids


if __name__ == '__main__':
    from ray.rllib.a3c.extended_evaluator import ShardA3CEvaluator, setup_sharded
    from ray.rllib.a3c import DEFAULT_CONFIG
    import gym
    ray.init()
    env_creator = lambda: gym.make("CartPole-v0")
    config = DEFAULT_CONFIG.copy()
    config["use_lstm"] = False
    config["ps_count"] = 2
    config["num_workers"] = 1
    logdir = "/tmp/shard"

    local_evaluator = ShardA3CEvaluator(env_creator, config, logdir)
    RemoteEAEvaluator = setup_sharded(len(local_evaluator.policy.get_weights()))
    remotes = [RemoteEAEvaluator.remote(env_creator, config, logdir)
                    for i in range(config["num_workers"])]
    optimizer = PSOptimizer(config, local_evaluator, remotes)

    optimizer.step()
