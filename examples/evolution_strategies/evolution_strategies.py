# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from collections import namedtuple
import gym
import numpy as np
import os
import pickle
import ray
import time

import optimizers
import policies
import tabular_logger as tlogger
import tf_util
import utils


Config = namedtuple("Config", [
    "l2coeff", "noise_stdev", "episodes_per_batch", "timesteps_per_batch",
    "calc_obstat_prob", "eval_prob", "snapshot_freq", "return_proc_mode",
    "episode_cutoff_mode"
])

Result = namedtuple("Result", [
    "noise_inds_n", "returns_n2", "sign_returns_n2", "lengths_n2",
    "eval_return", "eval_length", "ob_sum", "ob_sumsq", "ob_count", "no_noise"
])


@ray.remote
def create_shared_noise():
  """Create a large array of noise to be shared by all workers."""
  seed = 123
  count = 250000000
  noise = np.random.RandomState(seed).randn(count).astype(np.float32)
  return noise


class SharedNoiseTable(object):
  def __init__(self, noise):
    self.noise = noise
    assert self.noise.dtype == np.float32

  def get(self, i, dim):
    return self.noise[i:i + dim]

  def sample_index(self, stream, dim):
    return stream.randint(0, len(self.noise) - dim + 1)


@ray.remote
class Worker(object):
  def __init__(self, config, policy_params, env_name, noise,
               min_task_runtime=0.2):
    self.min_task_runtime = min_task_runtime
    self.config = config
    self.policy_params = policy_params
    self.noise = SharedNoiseTable(noise)

    self.env = gym.make(env_name)
    self.sess = utils.make_session(single_threaded=True)
    self.policy = policies.MujocoPolicy(self.env.observation_space,
                                        self.env.action_space,
                                        **policy_params)
    tf_util.initialize()

    self.rs = np.random.RandomState()

    assert self.policy.needs_ob_stat == (self.config.calc_obstat_prob != 0)

  def rollout_and_update_ob_stat(self, timestep_limit, task_ob_stat):
    if (self.policy.needs_ob_stat and self.config.calc_obstat_prob != 0 and
            self.rs.rand() < self.config.calc_obstat_prob):
      rollout_rews, rollout_len, obs = self.policy.rollout(
          self.env, timestep_limit=timestep_limit, save_obs=True,
          random_stream=self.rs)
      task_ob_stat.increment(obs.sum(axis=0), np.square(obs).sum(axis=0),
                             len(obs))
    else:
      rollout_rews, rollout_len = self.policy.rollout(
          self.env, timestep_limit=timestep_limit, random_stream=self.rs)
    return rollout_rews, rollout_len

  def no_op(self):
    return 1

  def do_rollouts(self, params, ob_mean, ob_std, timestep_limit=None):
    # Set the network weights.
    self.policy.set_trainable_flat(params)

    if self.policy.needs_ob_stat:
      self.policy.set_ob_stat(ob_mean, ob_std)

    if self.config.eval_prob != 0:
      raise NotImplementedError("Eval rollouts are not implemented.")

    noise_inds, returns, sign_returns, lengths = [], [], [], []
    # We set eps=0 because we're incrementing only.
    task_ob_stat = utils.RunningStat(self.env.observation_space.shape, eps=0)

    if np.random.uniform() < args.test_prob:
      self.policy.set_trainable_flat(params)
      rews_, len_ = self.rollout_and_update_ob_stat(timestep_limit,
                                                          task_ob_stat)
      # rews_neg, len_neg = self.rollout_and_update_ob_stat(timestep_limit,
      #                                                     task_ob_stat)
      # noise_inds.append(noise_idx)
      returns.append([rews_.sum()])
      sign_returns.append([np.sign(rews_).sum()])
      lengths.append([len_])
      return Result(
          noise_inds_n=np.array(noise_inds),
          returns_n2=np.array(returns, dtype=np.float32),
          sign_returns_n2=np.array(sign_returns, dtype=np.float32),
          lengths_n2=np.array(lengths, dtype=np.int32),
          eval_return=None,
          eval_length=None,
          ob_sum=(None if task_ob_stat.count == 0 else task_ob_stat.sum),
          ob_sumsq=(None if task_ob_stat.count == 0 else task_ob_stat.sumsq),
          ob_count=task_ob_stat.count,
          no_noise=True)

    # Perform some rollouts with noise.
    task_tstart = time.time()
    while (len(noise_inds) == 0 or
           time.time() - task_tstart < self.min_task_runtime):
      noise_idx = self.noise.sample_index(self.rs, self.policy.num_params)
      perturbation = self.config.noise_stdev * self.noise.get(
          noise_idx, self.policy.num_params)

      # These two sampling steps could be done in parallel on different actors
      # letting us update twice as frequently.
      self.policy.set_trainable_flat(params + perturbation)
      rews_pos, len_pos = self.rollout_and_update_ob_stat(timestep_limit,
                                                          task_ob_stat)

      self.policy.set_trainable_flat(params - perturbation)
      rews_neg, len_neg = self.rollout_and_update_ob_stat(timestep_limit,
                                                          task_ob_stat)

      noise_inds.append(noise_idx)
      returns.append([rews_pos.sum(), rews_neg.sum()])
      sign_returns.append([np.sign(rews_pos).sum(), np.sign(rews_neg).sum()])
      lengths.append([len_pos, len_neg])

      return Result(
          noise_inds_n=np.array(noise_inds),
          returns_n2=np.array(returns, dtype=np.float32),
          sign_returns_n2=np.array(sign_returns, dtype=np.float32),
          lengths_n2=np.array(lengths, dtype=np.int32),
          eval_return=None,
          eval_length=None,
          ob_sum=(None if task_ob_stat.count == 0 else task_ob_stat.sum),
          ob_sumsq=(None if task_ob_stat.count == 0 else task_ob_stat.sumsq),
          ob_count=task_ob_stat.count,
          no_noise=False)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train an RL agent on Pong.")
  parser.add_argument("--num-workers", default=10, type=int,
                      help=("The number of actors to create in aggregate "
                            "across the cluster."))
  parser.add_argument("--env-name", default="Pendulum-v0", type=str,
                      help="The name of the gym environment to use.")
  parser.add_argument("--stepsize", default=0.01, type=float,
                      help="The stepsize to use.")
  parser.add_argument("--redis-address", default=None, type=str,
                      help="The Redis address of the cluster.")
  parser.add_argument("--test-prob", default=None, type=float,
                      help="The probability of doing a test run.")
  parser.add_argument("--num-episodes", default=None, type=int,
                      help="The approximate number of episodes per update.")
  parser.add_argument("--warmup", default=0, type=int,
                      help="Warm up the plasma manager connections.")


  args = parser.parse_args()
  num_workers = args.num_workers
  env_name = args.env_name
  stepsize = args.stepsize

  import time
  filename = "es_{}_{}_{}_{}.pickle".format(args.num_workers, args.test_prob, args.num_episodes, time.time())
  results_file = open(filename, 'wb')

  ray.init(redis_address=args.redis_address,
           num_workers=(0 if args.redis_address is None else None))

  if args.warmup == 1:
    # For this to work, start each machine with 1 GPU.
    print("Running a bunch of tasks to warm up object manager connections")

    @ray.remote(num_gpus=1)
    def f():
      import time
      time.sleep(0.004)
      return 1

    @ray.remote(num_gpus=1)
    def g(x):
      import time
      time.sleep(0.004)
      return 1

    # These objects will be distributed around the cluster.
    ids = [f.remote() for _ in range(101)]

    i = 0
    for obj_id in ids:
      print(i)
      ray.get([g.remote(obj_id) for _ in range(101)])
      i += 1

    print("Finished running a bunch of tasks")
  else:
    print("Not warming up the object manager connections")

  config = Config(l2coeff=0.005,
                  noise_stdev=0.02,
                  episodes_per_batch=10000,
                  timesteps_per_batch=100000,
                  calc_obstat_prob=0.01,
                  eval_prob=0,
                  snapshot_freq=20,
                  return_proc_mode="centered_rank",
                  episode_cutoff_mode="env_default")

  policy_params = {
      "ac_bins": "continuous:",
      "ac_noise_std": 0.01,
      "nonlin_type": "tanh",
      "hidden_dims": [256, 256],
      "connection_type": "ff"
  }

  # Create the shared noise table.
  print("Creating shared noise table.")
  noise_id = create_shared_noise.remote()
  noise = SharedNoiseTable(ray.get(noise_id))

  # Create the actors.
  print("Creating actors.")
  workers = []
  for i in range(num_workers):
    workers.append(Worker.remote(config, policy_params, env_name, noise_id))
    if i % 100 == 0:
      print("i = ", i)

  env = gym.make(env_name)
  sess = utils.make_session(single_threaded=False)
  policy = policies.MujocoPolicy(env.observation_space, env.action_space,
                                 **policy_params)
  tf_util.initialize()
  optimizer = optimizers.Adam(policy, stepsize)

  ob_stat = utils.RunningStat(env.observation_space.shape, eps=1e-2)

  # Make sure the workers have started.
  print("Waiting for the workers to start.")
  ray.get([worker.no_op.remote() for worker in workers])
  print("The workers have started.")

  if args.warmup == 1:
    print("Sleeping for one min")
    time.sleep(60)
    print("Finished sleeping")

  episodes_so_far = 0
  timesteps_so_far = 0
  tstart = time.time()

  iteration = 0

  result_info = []

  while True:
    step_tstart = time.time()
    theta = policy.get_trainable_flat()
    assert theta.dtype == np.float32

    # Put the current policy weights in the object store.
    theta_id = ray.put(theta)

    # Divide by 2 because each one does 2.
    num_to_wait_for = int(np.ceil(args.num_episodes / 2))
    num_batches = int(np.ceil(args.num_episodes / 2 / len(workers)))

    # Use the actors to do rollouts, note that we pass in the ID of the policy
    # weights.
    rollout_ids = []
    for _ in range(num_batches):
        rollout_ids += [worker.do_rollouts.remote(
            theta_id,
            ob_stat.mean if policy.needs_ob_stat else None,
            ob_stat.std if policy.needs_ob_stat else None) for worker in workers]

    # Get the results of the rollouts.
    print("Submitting ", num_batches, " batches of ", len(workers))
    print("Waiting for ", num_to_wait_for)
    ready_ids, _ = ray.wait(rollout_ids, num_returns=num_to_wait_for)
    results = ray.get(ready_ids)

    curr_task_results = []
    ob_count_this_batch = 0

    test_returns = []
    test_lengths = []

    # Loop over the results
    for result in results:

      if result.no_noise:
        test_returns.extend(result.returns_n2)
        test_lengths.extend(result.lengths_n2)
        continue

      assert result.eval_length is None, "We aren't doing eval rollouts."
      assert result.noise_inds_n.ndim == 1
      assert result.returns_n2.shape == (len(result.noise_inds_n), 2)
      assert result.lengths_n2.shape == (len(result.noise_inds_n), 2)
      assert result.returns_n2.dtype == np.float32

      result_num_eps = result.lengths_n2.size
      result_num_timesteps = result.lengths_n2.sum()
      episodes_so_far += result_num_eps
      timesteps_so_far += result_num_timesteps

      curr_task_results.append(result)
      # Update ob stats.
      if policy.needs_ob_stat and result.ob_count > 0:
        ob_stat.increment(result.ob_sum, result.ob_sumsq, result.ob_count)
        ob_count_this_batch += result.ob_count

    print("NOISELESS RETURNS:", np.mean(test_returns))
    print("NOISELESS LENGTHS:", np.mean(test_lengths))

    # Assemble the results.
    noise_inds_n = np.concatenate([r.noise_inds_n for
                                   r in curr_task_results])
    returns_n2 = np.concatenate([r.returns_n2 for r in curr_task_results])
    lengths_n2 = np.concatenate([r.lengths_n2 for r in curr_task_results])
    assert noise_inds_n.shape[0] == returns_n2.shape[0] == lengths_n2.shape[0]
    # Process the returns.
    if config.return_proc_mode == "centered_rank":
      proc_returns_n2 = utils.compute_centered_ranks(returns_n2)
    else:
      raise NotImplementedError(config.return_proc_mode)

    # Compute and take a step.
    g, count = utils.batched_weighted_sum(
        proc_returns_n2[:, 0] - proc_returns_n2[:, 1],
        (noise.get(idx, policy.num_params) for idx in noise_inds_n),
        batch_size=500)
    g /= returns_n2.size
    assert (g.shape == (policy.num_params,) and g.dtype == np.float32 and
            count == len(noise_inds_n))
    update_ratio = optimizer.update(-g + config.l2coeff * theta)

    # Update ob stat (we're never running the policy in the master, but we
    # might be snapshotting the policy).
    if policy.needs_ob_stat:
      policy.set_ob_stat(ob_stat.mean, ob_stat.std)

    step_tend = time.time()
    tlogger.record_tabular("EpRewMean", returns_n2.mean())
    tlogger.record_tabular("EpRewStd", returns_n2.std())
    tlogger.record_tabular("EpLenMean", lengths_n2.mean())

    tlogger.record_tabular("Norm",
                           float(np.square(policy.get_trainable_flat()).sum()))
    tlogger.record_tabular("GradNorm", float(np.square(g).sum()))
    tlogger.record_tabular("UpdateRatio", float(update_ratio))

    tlogger.record_tabular("EpisodesThisIter", lengths_n2.size)
    tlogger.record_tabular("EpisodesSoFar", episodes_so_far)
    tlogger.record_tabular("TimestepsThisIter", lengths_n2.sum())
    tlogger.record_tabular("TimestepsSoFar", timesteps_so_far)

    tlogger.record_tabular("ObCount", ob_count_this_batch)

    tlogger.record_tabular("TimeElapsedThisIter", step_tend - step_tstart)
    tlogger.record_tabular("TimeElapsed", step_tend - tstart)
    tlogger.dump_tabular()

    result_info.append({
        "iteration": iteration,
        "noiseless returns": np.mean(test_returns),
        "noiseless lengths": np.mean(test_lengths),
        "timestamp": time.time(),
        "noisy returns": returns_n2.mean(),
        "noisy lengths": lengths_n2.mean()
    })

    if np.mean(test_returns) >= 6000:
      print("\n\nBreaking and storing results in ", filename, "\n\n")
      break

    iteration += 1

pickle.dump(result_info, results_file)
