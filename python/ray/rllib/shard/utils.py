import numpy as np

import ray
from ray.rllib.shard.gdoptimizers import Adam
from ray.rllib.utils.timer import TimerStat
from pandas import DataFrame

def shard(array, num):
    rets = np.array_split(array, num)
    return rets


class ParameterServer(object):
    def __init__(self, weight_shard: np.ndarray, config):

        self.params = weight_shard.copy()
        self.descent_optimizer = Adam(
            self.params, config.get("lr", 1e-4))
        print(self.params.shape)
        self.timers = {k: TimerStat() for k in ["update", "idle"]}
        self.timers["idle"].__enter__()

    def update_and_get_weights(self, grads):
        self.timers["idle"].__exit__(None, None, None)
        with self.timers["update"]:
            if type(grads) is list and len(grads) == 1:
                grads = grads[0]
            self.descent_optimizer.update(grads)

        self.timers["idle"].__enter__()
        return self.get_weights()

    def get_weights(self):
        return self.params

    def ip(self):
        return ray.services.get_node_ip_address()

    def pin(self, cpu_id):
        try:
            import psutil
            p = psutil.Process()
            p.cpu_affinity([cpu_id])
            print("Setting CPU Affinity to: ", cpu_id)
        except Exception as e:
            print(e)

    def stats(self):
        stats = {k + "_ms": round(1000 * v.mean, 3) for k, v in self.timers.items()}
        self.timers = {k: TimerStat() for k in ["update", "idle"]}
        self.timers["idle"].__enter__()
        return stats


class ShardedPS():
    def __init__(self, weights, config):
        self.ps_dict = {}
        if config["force"]:
            RemoteParamServer = ray.remote(num_gpus=1)(ParameterServer)
        else:
            RemoteParamServer = ray.remote(ParameterServer)

        for ps_id, weight_shard in enumerate(shard(weights, config["shards"])):
            self.ps_dict[ps_id] = RemoteParamServer.remote(weight_shard, config)
        self.iter = 0

    def update(self, sharded_grads: list):
        self.iter += 1
        weight_ids = []
        for ps_id, weight_shard in enumerate(sharded_grads):
            weight_ids.append(
                # for some reason this is hogging...
                self.ps_dict[ps_id].update_and_get_weights.remote(weight_shard))
        return weight_ids

    def get_weight_ids(self):
        return [self.ps_dict[ps_id].get_weights.remote() for ps_id in sorted(self.ps_dict)]

    def stats(self):
        df =  DataFrame(ray.get([ps.stats.remote() for ps in self.ps_dict.values()]))
        return dict(df.mean())


class PSClient(ShardedPS):
    def __init__(self, ps_dict):
        self.iter = 0
        self.ps_dict = ps_dict
