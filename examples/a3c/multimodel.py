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

@ray.actor
class A3C():
    def __init__(self, mid, num_workers=2, opt_type="adam", learning_rate=1e-4, env_name="PongDeterministic-v3", log_dir="/tmp/results/"):

        #inline defn needed
        @ray.actor(local=True)
        class Runner(object):
            """Actor object to start running simulation on workers.
                Gradient computation is also executed from this object."""
            def __init__(self, env_name, actor_id, logdir="./results/tf/", start=True):
                env = create_env(env_name)
                self.id = actor_id
                num_actions = env.action_space.n
                self.policy = LSTMPolicy(env.observation_space.shape, num_actions, actor_id)
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
                summary_writer = tf.summary.FileWriter(os.path.join(self.logdir, "agent_%d" % self.id))
                self.summary_writer = summary_writer
                self.runner.start_runner(self.policy.sess, summary_writer)

            def compute_gradient(self, params, tasksub):
                self.policy.set_weights(params)
                rollout = self.pull_batch_from_queue()
                batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)
                gradient = self.policy.get_gradients(batch)
                return gradient, None
        
        ## end inline ddef
        self.agents = [Runner(env_name, i, log_dir) for i in range(int(num_workers))]

        env = create_env(self.env_name)
        self.policy = LSTMPolicy(env.observation_space.shape, env.action_space.n, 0, opt_hparams={"learning_rate": learning_rate, "type": opt_type})
        if opt_type == "adam":
            assert self.policy.opt.get_name() == "Adam"

    def set_weights_with_optimizer(self, weights):
        self.policy.set_weights_with_optimizer(weights)

    def get_weights_with_optimizer(self):
        return self.policy.get_weights_with_optimizer()

    def set_weights(self, weights):
        self.policy.set_weights(weights)

    def get_weights(self):
        return self.policy.get_weights()

    def train(self, steps_max):
        parameters = self.policy.get_weights()
        gradient_list = [agent.compute_gradient(parameters, timestamp()) for agent in self.agents]
        steps = 0
        obs = 0

        while steps < steps_max:
            done_id, gradient_list = ray.wait(gradient_list)
            gradient, info = ray.get(done_id)[0]
            self.policy.model_update(gradient)
            parameters = self.policy.get_weights()
            steps += 1
            gradient_list.extend([self.agents[info["id"]].compute_gradient(parameters, timestamp())])

        return self.get_weights_with_optimizer(), timing

def model_averaging(params, stats=None):
    loader = defaultdict(list)
    for param_dict in params:
        for k, v in param_dict.items():
            loader[k].append(v)
    return {k: np.mean(v, axis=0) for k ,v in loader.items()}

def best_model(params, stats):
    mean = [m for m, s in stats]
    print(mean)
    best = np.argmax(mean)
    print("Choosing %d..." % best)
    return params[best]

def drop_half(params, stats):
    mean = [m for m, s in stats]
    idx = np.argsort(mean)
    idx = idx.repeat(2)[len(mean):] # take the top half
    return [params[i] for i in idx]

def evolution_1(params, stats):
    mean = [m for m, s in stats]
    idx_splt = np.split(np.argsort(mean), 2)
    top = idx_splt[1]
    bottom = np.random.choice(len(mean), size=len(idx_splt[0]))
    idx = np.concatenate([bottom, top])
    return [params[i] for i in idx]

def aggregation_function(val):
    if val == "average":
        return model_averaging
    if val == "drop_half":
        return drop_half
    if val == "best":
        return best_model
    if val == "evo1":
        return evolution_1 

def run_multimodel_experiment(exp_count=1, num_workers=10, opt_type="adam",
                    sync=10, learning_rate=1e-4, infostr="", 
                    addr_info=None, aggr_param="average",
                    load=""):
    aggregation = aggregation_function(aggr_param)
    SYNC = sync
    experiments = [A3C(i, num_workers, opt_type) for i in range(exp_count)]
    if load: # Load prewarmed weights
        new_params = load_weights(load)
        ray.get([e.set_weights(new_params) for i, e in enumerate(experiments)])

    new_params = ray.get(experiments[0].get_weights_with_optimizer())
    counter = 0
    itr = 0
    log = None
    while time.time() - _start < 1800:
        ray.get([e.set_weights_with_optimizer(param) for e, param in zip(experiments, new_params)]) # set weights on each A3C model
        return_vals = ray.get([e.train(SYNC) for i, e in enumerate(experiments)])
        params, information = zip(*return_vals)
        new_params = aggregation(params, stats)
    return params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the multi-model learning example.")
    parser.add_argument("--num-experiments", default=1, type=int, help="The number of training experiments")
    parser.add_argument("--runners", default=12, type=int, help="Number of simulations")
    parser.add_argument("--lr", default=1e-4, type=float, help="LearningRate")
    parser.add_argument("--type", default="adam", type=str, help="Type of Optimizer")
    parser.add_argument("--sync", default=10, type=int, help="Sync Step")
    parser.add_argument("--addr", default=None, type=str, help="The Redis address of the cluster.")
    parser.add_argument("--aggr", default="average", type=str, help="Aggregation Technique")
    parser.add_argument("--info", default="", type=str, help="Information for file name")
    parser.add_argument("--load", default="", type=str, help="Load pretrained weights")
    opts = parser.parse_args(sys.argv[1:])
    if opts.addr:
        address_info = ray.init(redirect_output=True, redis_address=opts.addr)
    else:
        address_info = ray.init(redirect_output=False, num_workers=1 )
    exp_results = run_multimodel_experiment(opts.num_experiments, 
                        num_workers=opts.runners, 
                        sync=opts.sync,
                        learning_rate=opts.lr,
                        opt_type=opts.type,
                        infostr=opts.info,
                        aggr_param=opts.aggr,
                        addr_info=address_info,
                        load=opts.load)
