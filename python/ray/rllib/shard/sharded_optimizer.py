import ray
import numpy as np
from ray.rllib.optimizers.optimizer import Optimizer
from ray.rllib.shard.gdoptimizers import Adam
from ray.rllib.shard.utils import shard, ShardedPS
from ray.rllib.utils.timer import TimerStat
from pandas import DataFrame


class PSOptimizer(Optimizer):

    def _init(self):
        self.timers = {k: TimerStat() for k in ["setup", "apply_call", "dispatch", "wait"]}
        weights = self.local_evaluator.get_flat()
        self.ps = ShardedPS(weights, self.config)
        self.workers = [Worker(remote_eval) for remote_eval in self.remote_evaluators]
        self.alive_workers = set()

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
                self.alive_workers.add(worker)
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
        stats["alive"] = len(self.alive_workers)


        df =  DataFrame(ray.get([evs.stats.remote() for evs in self.remote_evaluators]))
        stats.update(dict(df.mean()))

        self.alive_workers = set()
        self.timers = {k: TimerStat() for k in self.timers}
        return stats

    def stop(self):
        self.ps.stop()


class DriverlessPSOptimizer(PSOptimizer):

    def step(self):
        iters = int(self.config["grads_per_step"] / len(self.workers))
        ray.get([w.loop(iters) for w in self.workers])

    def create_ps_clients(self):
        return ray.get([w.create_ps_client(self.ps.ps_dict) for w in self.workers])


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

    def loop(self, iterations):
        return self._eval.loop.remote(iterations)

    def create_ps_client(self, ps_dict):
        list_of_ids, list_of_shards = zip(*ps_dict.items())
        return self._eval.create_ps_client.remote(list_of_ids, *list_of_shards)


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
