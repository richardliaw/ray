from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from collections import defaultdict
import numpy as np
from runner import RunnerThread, process_rollout
from LSTM import LSTMPolicy, RawLSTMPolicy
from weight_holder import WeightHolder
from FC import FCPolicy
import tensorflow as tf
import six.moves.queue as queue
import gym
import sys
import os
from datetime import datetime, timedelta
from misc import timestamp, time_string
from envs import create_env

POLICY = LSTMPolicy
@ray.actor
class Runner(object):
    """Actor object to start running simulation on workers.
        Gradient computation is also executed from this object."""
    def __init__(self, env_name, actor_id, logdir="results/", start=True):
        env = create_env(env_name)
        self.id = (actor_id)
        num_actions = env.action_space.n
        self.policy = POLICY(env.observation_space.shape, num_actions, actor_id)
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
        _start = timestamp()
        self.policy.set_weights(params)
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)
        gradient = self.policy.get_gradients(batch)
        _end = timestamp()
        info = {"id": self.id,
                "start_task": _start - extra,
                "time": _end -  _start,
                "end": _end,
                "size": len(batch.a)}
        return gradient, info
    
    def optimize(self, params):
        _start = timestamp()
        self.policy.set_weights(params)
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)
        gradient = self.policy.get_gradients(batch)
        self.policy.model_update(gradient)
        _end = timestamp()
        info = {"id": self.id,
                "start_task": _start - extra,
                "time": _end -  _start,
                "end": _end,
                "size": len(batch.a), 
                "results": batch.final}
        return parameter_delta(self.policy.get_weights())


def train(num_workers, step_size, env_name="PongDeterministic-v3"):
    env = create_env(env_name)
    policy = POLICY(env.observation_space.shape, env.action_space.n, 0)
    holder = WeightHolder(policy)
    
    agents = [Runner(env_name, i) for i in range(num_workers)]
    parameters = holder.get_weights()
    delta_list = [agent.optimize(parameters, timestamp()) for agent in agents]
    steps = 0
    obs = 0

    ## DEBUG
    timing = defaultdict(list)
    from csv import DictWriter
    log = None
    
    results = []

    while True:
        _start = timestamp()
        done_id, delta_list = ray.wait(delta_list)
        delta, info = ray.get(done_id)[0]
        _getwait = timestamp()
        holder.model_update(delta, step_size)

        _update = timestamp()
        parameters = holder.get_weights()
        _endget = timestamp()
        steps += 1
        obs += info["size"]
        delta_list.extend([agents[info["id"]].optimize(parameters, timestamp())])
        _endsubmit = timestamp()


        timing["Task"].append(info["time"])
        timing["Task_start"].append(info["start_task"])
        timing["Task_end"].append(_getwait - info["end"])
        timing["1.Wait"].append(_getwait - _start)
        timing["2.Update"].append(_update - _getwait)
        timing["3.Weights"].append(_endget - _update)
        timing["4.Submit"].append(_endsubmit - _endget)
        timing["5.Total"].append(_endsubmit - _start)
        results.extend(info["results"])
        if steps % 10 == 0:
            print("Model performance: Mean: %.4f | Std: %.4f" % (np.mean(results), np.std(results)))
            results = []

        if steps % 200 == 0:
            if log is None:
                log = DictWriter(open("./timing.csv", "w"), timing.keys())
                log.writeheader()
            print("####"*10 + " ".join(["%s: %f" % (k, np.mean(v)) for k, v in sorted(timing.items())]))
            log.writerow(timing)
            
            timing = defaultdict(list)
    return policy

if __name__ == '__main__':
    num_workers = int(sys.argv[1])
    step_size = float(sys.argv[2])
    ray.init(num_workers=1, redirect_output=True)
    train(num_workers, step_size, env_name="CartPole-v0")
