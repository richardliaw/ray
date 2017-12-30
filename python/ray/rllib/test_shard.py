from ray.rllib.a3c import DEFAULT_CONFIG
import sys
import gym
import argparse

import ray
import numpy as np
from ray.tune.result import TrainingResult, pretty_print
from ray.rllib.optimizers.optimizer import Optimizer
from ray.rllib.optimizers.sharded_optimizer import PSOptimizer
from ray.rllib.a3c.extended_evaluator import ShardA3CEvaluator, setup_sharded, shard

def get_metrics(remote_evaluators):
    episode_rewards = []
    episode_lengths = []
    metric_lists = [a.get_completed_rollout_metrics.remote()
                    for a in remote_evaluators]
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

    print(pretty_print(result))

    return result

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="Shard.")

# See also the base parser definition in ray/tune/config_parser.py
parser.add_argument("--redis-address", default=None, type=str,
                    help="The Redis address of the cluster.")
parser.add_argument("--num-workers", default=1, type=int,
                    help="Number of workers.")
parser.add_argument("--shards", default=1, type=int,
                    help="Number of GPUs to allocate to Ray.")
parser.add_argument("--force", default=False, type=bool,
                    help="Force actor placement.")

args = parser.parse_args(sys.argv[1:])
ray.init(redis_address=args.redis_address)

env_creator = lambda: gym.make("PongDeterministic-v0")
config = DEFAULT_CONFIG.copy()
config["use_lstm"] = True
config["ps_count"] = args.shards
config["num_workers"] = args.num_workers
config["force"] = args.force
config["preprocessing"]["dim"] = 42
logdir = "/tmp/shard"

local_evaluator = ShardA3CEvaluator(env_creator, config, logdir)
RemoteEAEvaluator = setup_sharded(config["ps_count"], force=config["force"])

remotes = [RemoteEAEvaluator.remote(
    env_creator, config, logdir,
    pin_id=0) for i in range(config["num_workers"])]
optimizer = PSOptimizer(config, local_evaluator, remotes)

for i in range(100):
    optimizer.step()
# import ipdb; ipdb.set_trace()
