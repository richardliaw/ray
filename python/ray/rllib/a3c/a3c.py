from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
import os

import ray
from ray.rllib.agent import Agent
from ray.rllib.a3c.envs import create_and_wrap
from ray.rllib.a3c.runner import RemoteRunner
from ray.rllib.a3c.sync_runner import RemoteSyncRunner
from ray.rllib.a3c.shared_model import SharedModel
from ray.rllib.a3c.shared_model_lstm import SharedModelLSTM
from ray.tune.result import TrainingResult


DEFAULT_CONFIG = {
    "num_workers": 4,
    "num_batches_per_iteration": 100,
    "batch_size": 10,
    "use_lstm": True,
    "model": {"grayscale": True,
              "zero_mean": False,
              "dim": 42,
              "channel_major": True}
}

def multinomial(vals):
    vals = np.array(vals)
    vals -= np.max(vals)
    v = np.exp(vals)
    p_vals = v / np.sum(v)
    return np.random.multinomial(n, p_vals)

class A3CAgent(Agent):
    _agent_name = "A3C"
    _default_config = DEFAULT_CONFIG

    def _init(self):
        self.env = create_and_wrap(self.env_creator, self.config["model"])
        if self.config["use_lstm"]:
            policy_cls = SharedModelLSTM
        else:
            policy_cls = SharedModel
        self.policy = policy_cls(
            self.env.observation_space.shape, self.env.action_space)
        self.agents = [
            RemoteSyncRunner.remote(self.env_creator, policy_cls, i,
                                self.config["batch_size"],
                                self.config["model"], self.logdir)
            for i in range(self.config["num_workers"])]
        self.parameters = self.policy.get_weights()
        self.iter = 0
        self.timing = {"grad_worker": 0, "grad_driver": 0}

    def _train(self):
        self.iter += 1
        if self.iter < 50:
            WORK = random.random() < 0.5
        else:
            WORK = multinomial([v for v in self.timing.values()])
        if WORK:
            t = time.time()
            self.grad_on_worker()
            dt = time.time() - t
            self.timing["grad_worker"] -= dt
        else:
            t = time.time()
            self.grad_on_driver()
            dt = time.time() - t
            self.timing["grad_driver"] -= dt
        res = self._fetch_metrics_from_workers()
        return res

    def grad_on_worker(self):
        gradient_list = [
            agent.compute_gradient.remote(self.parameters)
            for agent in self.agents]
        gradient_list = ray.get(gradient_list)
        p = [np.zeros_like(v) for v in gradient_list[0]]
        for grads in gradient_list:
            for i in enumerate(grads):
                p[i] += val
        self.policy.apply_gradients(p)

    def grad_on_driver(self):
        batches = [
            agent.set_sample.remote(self.parameters)
            for agent in self.agents]
        batches = ray.get(batches)
        import ipdb; ipdb.set_trace()
        # merge all batches together
        self.policy.model_update(p)

    def _fetch_metrics_from_workers(self):
        episode_rewards = []
        episode_lengths = []
        metric_lists = [
            a.get_completed_rollout_metrics.remote() for a in self.agents]
        for metrics in metric_lists:
            for episode in ray.get(metrics):
                episode_lengths.append(episode.episode_length)
                episode_rewards.append(episode.episode_reward)
        avg_reward = (
            np.mean(episode_rewards) if episode_rewards else float('nan'))
        avg_length = (
            np.mean(episode_lengths) if episode_lengths else float('nan'))
        timesteps = np.sum(episode_lengths) if episode_lengths else 0

        result = TrainingResult(
            episode_reward_mean=avg_reward,
            episode_len_mean=avg_length,
            timesteps_this_iter=timesteps,
            info={})

        return result

    def _save(self):
        checkpoint_path = os.path.join(
            self.logdir, "checkpoint-{}".format(self.iteration))
        objects = [self.parameters]
        pickle.dump(objects, open(checkpoint_path, "wb"))
        return checkpoint_path

    def _restore(self, checkpoint_path):
        objects = pickle.load(open(checkpoint_path, "rb"))
        self.parameters = objects[0]
        self.policy.set_weights(self.parameters)

    def compute_action(self, observation):
        actions = self.policy.compute_action(observation)
        return actions[0]
