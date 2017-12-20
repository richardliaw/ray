import ray
import numpy as np
from ray.rllib.optimizers.optimizer import Optimizer
from ray.rllib.a3c.extended_evaluator import ShardA3CEvaluator, setup_sharded, shard

@ray.remote
class ParameterServer(object):
    def __init__(self, weight_shard: np.ndarray, ps_id):
        self.ps_id = ps_id
        try:
            import psutil
            p = psutil.Process()
            p.cpu_affinity([ps_id])
            print("Setting CPU Affinity to: ", ps_id)
        except Exception as e:
            print(e)
            pass

        self.params = weight_shard.copy()
        print(self.params.shape)

    def update_and_get_weights(self, deltas):
        self.params += deltas
        return self.get_weights()

    def get_weights(self):
        return self.params

    def ip(self):
        return ray.services.get_node_ip_address()


class ShardedPS():
    def __init__(self, weights, ps_count):
        self.ps_dict = {}

        for ps_id, weight_shard in enumerate(shard(weights, ps_count)):
            self.ps_dict[ps_id] = ParameterServer.remote(weight_shard, ps_id)

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
        weight_ids = self.ps.get_weight_ids()
        worker_to_shard = {}
        shard_to_worker = {}
        for w in self.remote_evaluators:
            shards = w.compute_deltas.remote(*weight_ids)
            worker_to_shard[w] = shards
            shard_to_worker.update({t: w for t in shards})

        ray.wait(list(shard_to_worker), num_returns=len(shard_to_worker))

        for i in range(100):
            [done_shard], _ = ray.wait(list(shard_to_worker))
            w = shard_to_worker[done_shard]

            all_w_shards = worker_to_shard.pop(w)
            [shard_to_worker.pop(s) for s in all_w_shards]
            weight_ids = self.ps.update(all_w_shards)

            shards = w.compute_deltas.remote(*weight_ids)
            worker_to_shard[w] = shards
            shard_to_worker.update({t: w for t in shards})


if __name__ == '__main__':
    from ray.rllib.a3c import DEFAULT_CONFIG
    import gym
    ray.init()
    env_creator = lambda: gym.make("Pong-v0")
    config = DEFAULT_CONFIG.copy()
    # config["use_lstm"] = False
    config["ps_count"] = 10
    config["num_workers"] = 20
    logdir = "/tmp/shard"

    local_evaluator = ShardA3CEvaluator(env_creator, config, logdir)
    RemoteEAEvaluator = setup_sharded(config["ps_count"])

    remotes = [RemoteEAEvaluator.remote(
        env_creator, config, logdir,
        pin_id=(config["ps_count"] + i)) for i in range(config["num_workers"])]
    optimizer = PSOptimizer(config, local_evaluator, remotes)

    optimizer.step()
