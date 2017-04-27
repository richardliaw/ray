from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
import ray
import argparse
from collections import defaultdict
import numpy as np
from runner import RunnerThread, process_rollout
from LSTM import LSTMPolicy
from FC import FCPolicy
import tensorflow as tf
import six.moves.queue as queue
import gym
import sys
import os
from datetime import datetime, timedelta
from misc import timestamp, time_string, try_makedirs, load_weights, save_weights
from envs import create_env

@ray.actor
class Runner(object):
    """Actor object to start running simulation on workers.
        Gradient computation is also executed from this object."""
    def __init__(self, env_name, actor_id, logdir="./exp_results/", start=True):
        env = create_env(env_name)
        self.id = (actor_id)
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
        summary_writer = tf.summary.FileWriter(os.path.join(self.logdir, "agent_%d" % self.id))
        self.summary_writer = summary_writer
        self.runner.start_runner(self.policy.sess, summary_writer)

    def compute_gradient(self, params, extra):
        self.policy.set_weights(params)
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)
        gradient = self.policy.get_gradients(batch)
        return gradient, info


def train(num_workers, load="", save=False, env_name="PongDeterministic-v3"):
    FULL_START = timestamp()
    env = create_env(env_name)
    cfg = {"learning_rate": 1e-4, "type": "adam"}
    policy = LSTMPolicy(env.observation_space.shape, env.action_space.n, 0, opt_hparams=cfg)
    agents = [Runner(env_name, i) for i in range(num_workers)]

    if len(load):
        parameters = load_weights(load)
        print("LOADING WEIGHTS")
    else:
        parameters = policy.get_weights()

    gradient_list = [agent.compute_gradient(parameters, timestamp()) for agent in agents]
    steps = 0
    obs = 0

    counter = 0

    results = []

    while True:
        done_id, gradient_list = ray.wait(gradient_list)
        gradient, info = ray.get(done_id)[0]
        results.extend(info["result"])
        policy.model_update(gradient)
        parameters = policy.get_weights()

        gradient_list.extend([agents[info["id"]].compute_gradient(parameters, timestamp())])

    return policy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the multi-model learning example.")
    parser.add_argument("--load", default="", type=str, help="Pretrained-policy weights")
    parser.add_argument("--runners", default=12, type=int, help="Number of simulations")
    parser.add_argument("--save", default=False, type=bool, help="Save intermediate results")
    opts = parser.parse_args(sys.argv[1:])

    ray.init(num_workers=1 , redirect_output=True)
    train(opts.runners, load=opts.load, save=opts.save)
