from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
import os

import ray
from ray.rllib.agent import Agent
from ray.rllib.shard.sharded_optimizer import PSOptimizer, DriverlessPSOptimizer
from ray.rllib.shard.extended_evaluator import ShardA3CEvaluator, setup_sharded, shard
from ray.tune.result import TrainingResult, pretty_print


DEFAULT_CONFIG = {
    # Number of workers (excluding master)
    "num_workers": 4,
    # Size of rollout batch
    "batch_size": 10,
    # Use LSTM model - only applicable for image states
    "use_lstm": False,
    # Use PyTorch as backend - no LSTM support
    "use_pytorch": False,
    # Which observation filter to apply to the observation
    "observation_filter": "NoFilter",
    # Which reward filter to apply to the reward
    "reward_filter": "NoFilter",
    # Discount factor of MDP
    "gamma": 0.99,
    # GAE(gamma) parameter
    "lambda": 1.0,
    # Max global norm for each gradient calculated by worker
    "grad_clip": 40.0,
    # Value Function Loss coefficient
    "vf_loss_coeff": 0.5,
    # Entropy coefficient
    "entropy_coeff": -0.01,
    # Model and preprocessor options
    "model": {
        # (Image statespace) - Converts image to Channels = 1
        "grayscale": True,
        # (Image statespace) - Each pixel
        "zero_mean": False,
        # (Image statespace) - Converts image to (dim, dim, C)
        "dim": 80,
        # (Image statespace) - Converts image shape to (C, dim, dim)
        "channel_major": False
    },
    # Arguments to pass to the rllib optimizer
    "optimizer": {
        # Number of gradients applied for each `train` step
        "grads_per_step": 1000,
        # Number of shards
        "shards": 1,
        # Forces actor placement by specifying that each needs 1 GPU
        "force": False,
        # Learning rate
        "lr": 0.0001,
    },
    # Pins actors to cores
    "pin": False,
    # This is only here to appease a3c/tfpolicy
    "lr": 0.0001,
    "selfdriving": False
}


class ShardedAgent(Agent):
    _agent_name = "Sharded"
    _default_config = DEFAULT_CONFIG
    _allow_unknown_subkeys = ["model", "optimizer"]

    def _init(self):
        self.local_evaluator = ShardA3CEvaluator(
            self.registry, self.env_creator, self.config, self.logdir, start_sampler=False)
        RemoteEAEvaluator = setup_sharded(
            self.config["optimizer"]["shards"],
            force=self.config["optimizer"]["force"])

        self.remote_evaluators = [RemoteEAEvaluator.remote(
            self.registry, self.env_creator, self.config, self.logdir)
            for i in range(self.config["num_workers"])]

        if self.config["selfdriving"]:
            opt_class = DriverlessPSOptimizer
        else:
            opt_class = PSOptimizer
        self.optimizer = opt_classt(
            self.config["optimizer"], self.local_evaluator,
            self.remote_evaluators)

        if self.config.get("pin"):
            [actor.pin.remote(i) for i, actor in
                enumerate(list(self.optimizer.ps.ps_dict.values()) + self.remote_evaluators)]

    def _train(self):
        self.optimizer.step()
        # FilterManager.synchronize(
        #     self.local_evaluator.filters, self.remote_evaluators)
        res = self._fetch_metrics_from_remote_evaluators()
        return res

    def _fetch_metrics_from_remote_evaluators(self):
        episode_rewards = []
        episode_lengths = []
        metric_lists = [a.get_completed_rollout_metrics.remote()
                        for a in self.remote_evaluators]
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
            mean_loss=np.max(episode_rewards),
            info={**self.optimizer.stats()})
        return result

    def _save(self):
        raise NotImplementedError

    def _restore(self, checkpoint_path):
        raise NotImplementedError

    def compute_action(self, observation):
        raise NotImplementedError
