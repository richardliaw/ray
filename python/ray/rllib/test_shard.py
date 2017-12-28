from ray.rllib.a3c import DEFAULT_CONFIG
import sys
import gym
import argparse


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

args = parser.parse_args(sys.argv[1:])
ray.init(redis_address=args.redis_address)

env_creator = lambda: gym.make("Pong-v0")
config = DEFAULT_CONFIG.copy()
# config["use_lstm"] = False
config["ps_count"] = args.shards
config["num_workers"] = args.num_workers
logdir = "/tmp/shard"

local_evaluator = ShardA3CEvaluator(env_creator, config, logdir)
RemoteEAEvaluator = setup_sharded(config["ps_count"])

remotes = [RemoteEAEvaluator.remote(
    env_creator, config, logdir,
    pin_id=(config["ps_count"] + i)) for i in range(config["num_workers"])]
optimizer = PSOptimizer(config, local_evaluator, remotes)

optimizer.step()
import ipdb; ipdb.set_trace()
