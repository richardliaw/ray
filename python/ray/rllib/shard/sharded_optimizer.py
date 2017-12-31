import ray
import numpy as np
from ray.rllib.optimizers.optimizer import Optimizer
from ray.rllib.shard.extended_evaluator import ShardA3CEvaluator, setup_sharded, shard

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
        if type(deltas) is list and len(deltas) == 1:
            deltas = deltas[0]
        self.params += deltas
        return self.get_weights()

    def get_weights(self):
        return self.params

    def ip(self):
        return ray.services.get_node_ip_address()


class ShardedPS():
    def __init__(self, weights, num_shards, force):
        self.ps_dict = {}

        for ps_id, weight_shard in enumerate(shard(weights, num_shards)):
            if force:
                RemoteParamServer = ray.remote(num_gpus=1)(ParameterServer)
            else:
                RemoteParamServer = ray.remote(ParameterServer)
            self.ps_dict[ps_id] = RemoteParamServer.remote(weight_shard, ps_id)
        self.iter = 0

    def update(self, sharded_deltas: list):
        self.iter += 1
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
        self.ps = ShardedPS(weights, self.config["shards"], self.config["force"])
        self.workers = [Worker(remote_eval) for remote_eval in self.remote_evaluators]

    def step(self):
        # send deltas to parameter servers
        weight_ids = self.ps.get_weight_ids()
        for w in self.workers:
            new_deltas = w.compute_deltas(weight_ids)
            w.track_deltas(new_deltas)

        WorkerQ.wait_for_all(self.workers)

        for i in range(self.config["grads_per_step"]):
            worker = WorkerQ.next_completed(self.workers)
            new_weights = self.ps.update(worker.deltas)
            new_deltas = worker.compute_deltas(new_weights)
            worker.weight_iter = self.ps.iter
            worker.track_deltas(new_deltas)


class Worker():
    """Wrapper class to Extended Evaluators"""
    def __init__(self, evaluator):
        self._eval = evaluator
        self.deltas = []  # ObjectIDs
        self.weight_iter = 0

    def track_deltas(self, new_deltas):
        if type(new_deltas) is list:
            self.deltas = new_deltas
        else:
            if len(self.deltas):
                self.deltas.pop(0)
            self.deltas.append(new_deltas)

    def compute_deltas(self, weight_list: list):
        return self._eval.compute_deltas.remote(*weight_list)


class WorkerQ():
    @staticmethod
    def next_completed(workers):
        obj_to_worker = {k: w for w in workers for k in w.deltas}
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
        all_objs = [k for w in workers for k in w.deltas]
        ray.wait(all_objs, num_returns=len(all_objs))

