import sys
import gym
import argparse

import ray
import numpy as np
from ray.tune.result import TrainingResult, pretty_print
from ray.rllib.shard.shardedagent import ShardedAgent, DEFAULT_CONFIG

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
parser.add_argument("--pin", default=False, type=bool,
                    help="Pin Actors.")
parser.add_argument("--force", default=False, type=bool,
                    help="Force actor placement.")

args = parser.parse_args(sys.argv[1:])
ray.init(redis_address=args.redis_address)

config = DEFAULT_CONFIG.copy()
config["use_lstm"] = True
config["pin"] = args.pin
config["num_workers"] = args.num_workers
config["optimizer"]["shards"] = args.shards
config["optimizer"]["force"] = args.force
config["model"]["dim"] = 42

agent = ShardedAgent(config, "PongDeterministic-v0")
for i in range(5):
    res = agent.train()
