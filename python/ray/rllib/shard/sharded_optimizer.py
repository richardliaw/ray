import ray
import numpy as np
from ray.rllib.optimizers.optimizer import Optimizer
from ray.rllib.shard.extended_evaluator import ShardA3CEvaluator, setup_sharded, shard
from ray.rllib.shard.gdoptimizers import Adam
from ray.rllib.utils.timer import TimerStat
from pandas import DataFrame


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
                self.ps_dict[ps_id].update_and_get_weights.remote(weight_shard))
        return weight_ids

    def get_weight_ids(self):
        return [self.ps_dict[ps_id].get_weights.remote() for ps_id in sorted(self.ps_dict)]

    def stats(self):
        df =  DataFrame(ray.get([ps.stats.remote() for ps in self.ps_dict.values()]))
        return dict(df.mean())


class PSOptimizer(Optimizer):

    def _init(self):
        self.timers = {k: TimerStat() for k in ["setup", "apply_call", "dispatch", "wait"]}
        weights = self.local_evaluator.get_flat()
        self.ps = ShardedPS(weights, self.config)
        self.workers = [Worker(remote_eval) for remote_eval in self.remote_evaluators]

    def step(self):
        # send grads to parameter servers
        with self.timers["setup"]:
            if any(len(w.grads) == 0 for w in self.workers):
                weight_ids = self.ps.get_weight_ids()
                for w in self.workers:
                    if not w.grads:
                        new_grads = w.compute_flat_grad(weight_ids)
                        w.track_grads(new_grads)
                WorkerQ.wait_for_all(self.workers)

        for i in range(self.config["grads_per_step"]):
            with self.timers["wait"]:
                worker = WorkerQ.next_completed(self.workers)
                # try just dropping things that are too late

            with self.timers["apply_call"]:
                new_weights = self.ps.update(worker.grads)

            with self.timers["dispatch"]:
                new_grads = worker.compute_flat_grad(new_weights)
                worker.weight_iter = self.ps.iter
                worker.track_grads(new_grads)

    def stats(self):
        stats = {k + "_ms": round(1000 * v.mean, 3) for k, v in self.timers.items()}
        stats.update(self.ps.stats())
        self.timers = {k: TimerStat() for k in self.timers}
        return stats


class Worker():
    """Wrapper class to Extended Evaluators"""
    def __init__(self, evaluator):
        self._eval = evaluator
        self.grads = []  # ObjectIDs
        self.weight_iter = 0

    def track_grads(self, new_grads):
        if type(new_grads) is list:
            self.grads = new_grads
        else:
            if len(self.grads):
                self.grads.pop(0)
            self.grads.append(new_grads)

    def compute_flat_grad(self, weight_list: list):
        return self._eval.compute_flat_grad.remote(*weight_list)


class WorkerQ():
    @staticmethod
    def next_completed(workers):
        obj_to_worker = {k: w for w in workers for k in w.grads}
        [done_obj], _ = ray.wait(list(obj_to_worker))
        return obj_to_worker[done_obj]

    @staticmethod
    def circular(workers):
        # This reduces the need to worry about arbitrarily stale updates
        # no need to wait since deterministic
        worker = workers.pop(0)
        workers.append(worker)
        return worker

    @staticmethod
    def wait_for_all(workers):
        all_objs = [k for w in workers for k in w.grads]
        ray.wait(all_objs, num_returns=len(all_objs))

