import argparse
import numpy as np
import json
from collections import defaultdict
import time
from datetime import timedelta
import ray
import tensorflow as tf
from driver import Runner
from six.moves import queue
from runner import RunnerThread, process_rollout, UpdateThread
import sys, os
from misc import *
from envs import create_env
from LSTM import LSTMPolicy
from FC import FCPolicy
from csv import DictWriter
import threading
BATCH = 20

def model_averaging(params):
    loader = defaultdict(list)
    for param_dict in params:
        for k, v in param_dict.items():
            loader[k].append(v)
    return {k: np.mean(v, axis=0) for k ,v in loader.items()}

def best_model(params, stats):
    mean = [m - s for m, s in stats]
    best = np.argmax(mean)
    print("Choosing %d..." % best)
    return params[best]

@ray.remote
def run_multimodel_experiment(exp_count=1, num_workers=10, adam=False,
                    sync=10, learning_rate=1e-4, infostr="", addr_info=None):

    @ray.actor
    class Training():
        def __init__(self, num_workers=2, adam=False, learning_rate=1e-4, env_name="CartPole-v0", log_dir="/tmp/results/"):
            assert type(adam) == bool
            try:
                os.makedirs(log_dir)
            except Exception as e:
                pass
            self.env_name = env_name
            self.log_dir = log_dir
            self.driver_node = ray.services.get_node_ip_address()
    
            #inline defn needed
            @ray.actor(local=True)
            class Runner(object):
                """Actor object to start running simulation on workers.
                    Gradient computation is also executed from this object."""
                def __init__(self, env_name, actor_id, logdir="./results/tf/", start=True):
                    env = create_env(env_name)
                    self.id = actor_id
                    num_actions = env.action_space.n
                    self.policy = FCPolicy(env.observation_space.shape, num_actions, actor_id)
                    self.runner = RunnerThread(env, self.policy, BATCH)
                    self.env = env
                    self.logdir = logdir
                    if start:
                        self.start()
    
                def pull_batch_from_queue(self):
                    """ self explanatory:  take a rollout from the queue of the thread runner. """
                    rollout = self.runner.queue.get(timeout=600.0)
                    while not rollout.terminal:
                        try:
                            rollout.extend(self.runner.queue.get_nowait())
                        except queue.Empty:
                            break
                    return rollout
    
                def start(self):
                    print("starting runner")
                    summary_writer = tf.summary.FileWriter(os.path.join(self.logdir, "agent_%d" % self.id))
                    self.summary_writer = summary_writer
                    self.runner.start_runner(self.policy.sess, summary_writer)
    
                def compute_gradient(self, params):
                    _start = timestamp()
                    self.policy.set_weights(params)
                    rollout = self.pull_batch_from_queue()
                    batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)
                    gradient = self.policy.get_gradients(batch)
                    info = {"id": self.id,
                            "size": len(batch.a),
                            "node": ray.services.get_node_ip_address(),
                            "start": _start,
                            "end": timestamp(),
                            "results": rollout.final }
                    return gradient, info
            ## end inline ddef
            self.agents = [Runner(env_name, i, log_dir) for i in range(int(num_workers))]
    
            env = create_env(self.env_name)
            self.policy = FCPolicy(env.observation_space.shape, env.action_space.n, 0, opt_hparams={"learning_rate": learning_rate, "adam": adam})
            if adam:
                assert self.policy.optimizer.get_name() == "Adam"
      
    
        def get_log_dir(self):
            return self.log_dir
        
        def get_driver_node(self):
            return self.driver_node
    
        def train(self, steps_max):
            parameters = self.policy.get_weights()
            gradient_list = [agent.compute_gradient(parameters) for agent in self.agents]
            steps = 0
            obs = 0
            results = []
    
            while steps < steps_max:
                # _start = timestamp()
                done_id, gradient_list = ray.wait(gradient_list)
                gradient, info = ray.get(done_id)[0]
                if info['results']:
                    results.extend(info['results'])
                # _getwait = timestamp()
                self.policy.model_update(gradient)
                # _update = timestamp()
                parameters = self.policy.get_weights()
                # _endget = timestamp()
                steps += 1
                obs += info["size"]
                gradient_list.extend([self.agents[info["id"]].compute_gradient(parameters)])
            return self.policy.get_weights(), results
    
        def set_weights(self, weights):
            print("Setting weights...")
            self.policy.set_weights(weights)
    
        def get_weights(self):
            return self.policy.get_weights()
    
    
        def async_train(self, steps_max, addr_info):
            env = create_env(self.env_name)
            self.policy = FCPolicy(env.observation_space.shape, env.action_space.n, 0)
            parameters = self.policy.get_weights()
            gradient_list = [agent.compute_gradient(parameters) for agent in self.agents]
            self.new_param_q = queue.Queue()
            self.async_task_q = queue.Queue(6) # arbitrary
            update_threads = [UpdateThread(self.policy, 
                                            self.new_param_q,
                                            self.async_task_q,
                                            addr_info) for i in range(1)] # arbitrary
            steps = 0
            info_list = []
            timing = defaultdict(list)
            while steps < steps_max:
                _start = timestamp()
                done_id, gradient_list = ray.wait(gradient_list)
                _endwait = timestamp()
                self.async_update(done_id)
                steps += 1
                _endasync = timestamp()
                gradient_list.extend(self.pull_submitted(len(gradient_list) == 0))
                _end = timestamp()
                timing["Wait"].append(_endwait - _start)
                timing["Async"].append(_endasync - _endwait)
                timing["Submit"].append( _end - _endasync)
                timing["Total"].append(_end - _start)
                if steps % 200 == 0:
                    print("## #" * 10 + str(steps) + " ".join(["%s Time: %f" % (k, np.mean(v)) 
                                        for k, v in sorted(timing.items())]))
                timing = defaultdict(list)
            return info_list
    
        def async_update(self, done_id):
            grad, info = ray.get(done_id)[0]
            self.async_task_q.put((info["id"], grad), timeout=600.0)
    
            # uthread = UpdateThread(done_id, self.policy, self.new_param_q, self.agents)
            # uthread.starter()
    
        def pull_submitted(self, block):
            """ Pulls a new object id off queue, starts the task """
            params = []
            if block:
                params.append(self.new_param_q.get())
            while True:
                try:
                    params.append(self.new_param_q.get_nowait())
                except queue.Empty:
                    break
            submitted = [self.agents[actor_id].compute_gradient(ray.local_scheduler.ObjectID(param_str) )
                            for actor_id, param_str in params]
            return submitted
        
        def get_gradient(self):
            pass
    
    SYNC = sync
    _start = time.time()
    experiments = [Training(num_workers, adam) for i in range(exp_count)]
    all_info = defaultdict(list)
    all_info["exp_count"] = exp_count
    all_info["sync"] = SYNC
    all_info["workers"] = num_workers
    all_info["batch"] = BATCH

    new_params = ray.get(experiments[0].get_weights())
    counter = 0
    while True:
        ray.get([e.set_weights(new_params) for i, e in enumerate(experiments)])
        print("Set weights")
        return_vals = ray.get([e.train(SYNC) for i, e in enumerate(experiments)])
        params, results = zip(*return_vals)
        stats = [(np.mean(x), np.std(x)) for x in results]
        all_info["stats"].append(stats)
        all_info["TS"].append((time.time() - _start))

        if np.mean(stats, axis=0)[0] > 190:
            counter += 1
        else: counter = 0
        if counter > 4 or (time.time() - _start) > 210:
            break
        time_str = str(timedelta(seconds=time.time() - _start))
        print("Time elapsed: " + time_str)

        print("Model performance: \n" + "\n".join(["%d -- Mean: %.4f | Std: %.4f" % (i, m, s) for i, (m, s) in enumerate(stats)])) 
        new_params = model_averaging(params)
        # new_params = best_model(params, stats)
    fdir = "./results/e{0}w{1}_lr{2}_sync{3}/".format(exp_count, 
                                                      num_workers,
                                                      learning_rate,
                                                      SYNC)
    if adam:
        fdir = fdir[:-1] + "adam/"
    all_info["exp_string"] = fdir
    return all_info

def save_results(exp_results):
    fdir = exp_results["exp_string"]
    try:
        os.makedirs(fdir)
    except Exception:
        pass
    with open(fdir + time.time() + ".json", "w") as f:
        json.dump(exp_results, f)
    print("Done")

def main():
    ray.init(num_workers=1)
    import ipdb; ipdb.set_trace()
    all_experiments = []
    for repeat in range(3):
        for sync in [10, 30, 100]:
            for learning_rate in [10**(-x) for x in range(2, 5)]:
                all_experiments.append(run_multimodel_experiment.remote())

    while len(all_experiments):
        done, all_experiments = ray.wait(all_experiments)
        results = ray.get(done)
        import ipdb; ipdb.set_trace()
        save_results(results)



if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Run the multi-model learning example.")
    # parser.add_argument("--num-experiments", default=1, type=int, help="The number of training experiments")
    # parser.add_argument("--runners", default=6, type=int, help="Number of simulations")
    # parser.add_argument("--lr", default=1e-5, type=float, help="LearningRate")
    # parser.add_argument("--adam", default=False, type=bool, help="ADAM")
    # parser.add_argument("--sync", default=10, type=int, help="Sync Step")
    # parser.add_argument("--addr", default=None, type=str, help="The Redis address of the cluster.")
    # parser.add_argument("--info", default="", type=str, help="Information for file name")
    main()
    # opts = parser.parse_args(sys.argv[1:])
    # if opts.addr:
    #     address_info = ray.init(redirect_output=True, redis_address=opts.addr)
    # else:
    #     address_info = ray.init(redirect_output=True)
    # address_info["store_socket_name"] = address_info["object_store_addresses"][0].name
    # address_info["manager_socket_name"] = address_info["object_store_addresses"][0].manager_name
    # address_info["local_scheduler_socket_name"] = address_info["local_scheduler_socket_names"][0]
    # ray.register_class(ray.services.ObjectStoreAddress)
    # # addr_info_id = ray.put(address_info)
    # exp_results = run_multimodel_experiment(opts.num_experiments, 
    #                     num_workers=opts.runners, 
    #                     sync=opts.sync,
    #                     learning_rate=opts.lr,
    #                     adam=opts.adam,
    #                     infostr=opts.info,
    #                     addr_info=address_info)
    # save_results(exp_results)
