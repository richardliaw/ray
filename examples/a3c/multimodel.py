import argparse
import numpy as np
import json
from collections import defaultdict
import ray
import tensorflow as tf
from driver import Runner
from six.moves import queue
from runner import RunnerThread, process_rollout, UpdateThread
import sys, os
from misc import *
from envs import create_env
from LSTM import LSTMPolicy
from csv import DictWriter
import threading

@ray.actor
class Training():
    def __init__(self, num_workers=2, env_name="PongDeterministic-v3", log_dir="/tmp/results/"):
        try:
            os.makedirs(log_dir)
        except Exception as e:
            pass
        self.env_name = env_name
        self.log_dir = log_dir
        self.driver_node = ray.services.get_node_ip_address()

        #inline defn needed
        @ray.actor
        class Runner(object):
            """Actor object to start running simulation on workers.
                Gradient computation is also executed from this object."""
            def __init__(self, env_name, actor_id, logdir="results/", start=True):
                env = create_env(env_name)
                self.id = actor_id
                num_actions = env.action_space.n
                self.policy = LSTMPolicy(env.observation_space.shape, num_actions, actor_id)
                self.runner = RunnerThread(env, self.policy, 20)
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
                        "end": timestamp(), }
                return gradient, info
        ## end inline ddef
        self.agents = [Runner(env_name, i, log_dir) for i in range(int(num_workers))]

    def get_log_dir(self):
        return self.log_dir
    
    def get_driver_node(self):
        return self.driver_node
    
    def async_train(self, steps_max, addr_info):
        env = create_env(self.env_name)
        self.policy = LSTMPolicy(env.observation_space.shape, env.action_space.n, 0)
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


def manager_begin(exp_count=1, num_workers=10, infostr="", addr_info=None):
    experiments = [Training(num_workers) for i in range(exp_count)]
    all_info = []
    headers = ["id", "size", "node", "start", "end"]
    for i, e in enumerate(experiments):
        print("Starting new exp..")
        stats = defaultdict(lambda: defaultdict(int))
        info_list = e.train(5 * 1e7, addr_info)
        info_list = ray.get(info_list)
        log_dir = ray.get(e.get_log_dir())
        with open(log_dir + "timing_" + infostr + "_%d.csv" % i, "w") as f:
            writer = DictWriter(f, headers)
            writer.writeheader()
            writer.writerows(info_list)
        for info in info_list:
            idx = info['id']
            stats[idx]['node'] = info['node']
            if stats[idx]['start'] == 0:
                stats[idx]['start'] = info['start']
            stats[idx]['end'] = info['end']
            stats[idx]['cpu_time'] += (info['end'] - info['start'])
            stats[idx]['obs'] += info['size']
            stats[idx]['count'] += 1
        stats["driver_node"] = ray.get(e.get_driver_node())
        stats = json.loads(json.dumps(stats))
        all_info.append(stats)
    import pickle
    fname = os.path.join(log_dir, "stats_{0}.pkl".format(infostr))
    with open(fname, "wb") as f:
        pickle.dump(all_info, f)
    print("Done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the multi-model learning example.")
    parser.add_argument("--num-experiments", default=1, type=int, help="The number of training experiments")
    parser.add_argument("--runners", default=10, type=int, help="Number of simulations")
    parser.add_argument("--addr", default=None, type=str, help="The Redis address of the cluster.")
    parser.add_argument("--info", default="", type=str, help="Information for file name")
    opts = parser.parse_args(sys.argv[1:])
    if opts.addr:
        address_info = ray.init(redis_address=opts.addr, num_cpus=5, num_workers=5)
    else:
        address_info = ray.init(num_cpus=5, num_workers=5)
    address_info["store_socket_name"] = address_info["object_store_addresses"][0].name
    address_info["manager_socket_name"] = address_info["object_store_addresses"][0].manager_name
    address_info["local_scheduler_socket_name"] = address_info["local_scheduler_socket_names"][0]
    ray.register_class(ray.services.ObjectStoreAddress)
    # addr_info_id = ray.put(address_info)
    exp = manager_begin(opts.num_experiments, 
                        num_workers=opts.runners, 
                        infostr=opts.info,
                        addr_info=address_info)
