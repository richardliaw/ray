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

# Result = namedtuple("Result", [
#     "noise_inds_n", "returns_n2", "sign_returns_n2", "lengths_n2",
#     "eval_return", "eval_length", "ob_sum", "ob_sumsq", "ob_count", "no_noise"
# ])


@ray.remote
def create_shared_noise():
  """Create a large array of noise to be shared by all workers."""
  seed = 123
  #count = 250000000
  count = 2500000
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

def _process_subworker_timing(tm):
  return {"SUB_launch_to_start": tm["start"] - tm["submit"],
          "SUB_rollouts": np.mean(tm["ro_duration"]),
          "SUB_setup": tm["setup"] - tm["start"],
          "SUB_duration": tm["end"] - tm["start"],
          "SUB_time_till_collection": time.time() - tm["end"]}

def _process_ma_timing(tm):
  tm["compute_g"] = tm["end"] - tm["process_returns"]
  tm["process_returns"] = tm["process_returns"] - tm["process_obstats"]
  tm["process_obstats"] = tm["process_obstats"] - tm["hier_rollouts"]
  tm["hierr_o_duration"] = tm["hier_rollouts"] - tm["start"]
  tm["launch_to_start"] = tm["start"] - tm["submit"]
  tm["time_till_collection"] = time.time() - tm["end"]
  return tm

def _average_dicts(list_dicts):
  ret = {}
  template = list_dicts[0]
  for k in template.keys():
    ret[k] = np.mean([d[k] for d in list_dicts])
  return ret

@ray.remote
class Worker(object):
  def __init__(self, config, policy_params, env_name, noise,
               min_task_runtime=0.2, num_episode_pairs_per_worker=2, pin_workers=False):
    # Pin this worker to a core.
    if pin_workers:
        key = "pin" + ray.worker.global_worker.node_ip_address
        # The first value returned by incr is 1.
        index = (ray.worker.global_worker.redis_client.incr(key) - 1)
        import psutil
        p = psutil.Process()
        p.cpu_affinity([index])
        print("Pinning to core ", index)


    self.min_task_runtime = min_task_runtime
    self.num_episode_pairs_per_worker = num_episode_pairs_per_worker

    # if 'time_per_batch' in config and 'episodes_per_batch' in config:
    #   raise Exception("Cannot use both time_per_batch and episodes_per_batch")

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

  def do_rollouts(self, params, ob_mean, ob_std, timestep_limit=None, submit=None):
    # Set the network weights.
    timing = {}
    timing["submit"] = submit
    timing["start"] = time.time()
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
      return {
          "noise_inds_n":np.array(noise_inds),
          "returns_n2":np.array(returns, dtype=np.float32),
          "sign_returns_n2":np.array(sign_returns, dtype=np.float32),
          "lengths_n2":np.array(lengths, dtype=np.int32),
          "eval_return":None,
          "eval_length":None,
          "ob_sum":(None if task_ob_stat.count == 0 else task_ob_stat.sum),
          "ob_sumsq":(None if task_ob_stat.count == 0 else task_ob_stat.sumsq),
          "ob_count":task_ob_stat.count,
          "no_noise":True
          }

    # Perform some rollouts with noise.
    timing["setup"] = time.time()
    timing["rollout_start"] = time.time()
    timing["ro_duration"] = []
    task_tstart = time.time()
    while (len(noise_inds) < 2 * self.num_episode_pairs_per_worker or
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
      timing["ro_duration"].append(time.time() - timing["rollout_start"])
    timing["end"] = time.time()

    return {
      "noise_inds_n":np.array(noise_inds),
      "returns_n2":np.array(returns, dtype=np.float32),
      "sign_returns_n2":np.array(sign_returns, dtype=np.float32),
      "lengths_n2":np.array(lengths, dtype=np.int32),
      "eval_return":None,
      "eval_length":None,
      "ob_sum":(None if task_ob_stat.count == 0 else task_ob_stat.sum),
      "ob_sumsq":(None if task_ob_stat.count == 0 else task_ob_stat.sumsq),
      "ob_count":task_ob_stat.count,
      "no_noise":False,
      "timing": timing
      }

@ray.remote
class MasterWorker(object):
    def __init__(self, num_workers_to_create, config, policy_params, env_name, noise_id,
                 min_task_runtime=0.2, num_episode_pairs_per_worker=2, pin_workers=False):
        self.workers = [Worker.remote(config, policy_params, env_name, noise_id[0],
                                      min_task_runtime=args.min_task_runtime,
                                      num_episode_pairs_per_worker=num_episode_pairs_per_worker,
                                      pin_workers=pin_workers)
                        for _ in range(num_workers_to_create)]

    def no_op(self):
        return ray.get([w.no_op.remote() for w in self.workers])

    def do_rollouts_and_return_update(self, params, policy_num_params, ob_mean, ob_std, timestep_limit=None, submit=None):
        timing = {}
        timing["submit"] = submit
        timing["start"] = time.time()
        results = ray.get([w.do_rollouts.remote(params[0], ob_mean, ob_std, timestep_limit=timestep_limit, submit=time.time())
                           for w in self.workers])
        timing["hier_rollouts"] = time.time()

        curr_task_results = []
        ob_count_this_batch = 0

        test_returns = []
        test_lengths = []

        ob_sum = None
        ob_sumsq = None
        ob_count = None
        worker_timing = []

        # Loop over the results
        for result in results:

          if result['no_noise']:
            test_returns.extend(result['returns_n2'])
            test_lengths.extend(result['lengths_n2'])
            continue

          assert result['eval_length'] is None, "We aren't doing eval rollouts."
          assert result['noise_inds_n'].ndim == 1
          assert result['returns_n2'].shape == (len(result['noise_inds_n']), 2)
          assert result['lengths_n2'].shape == (len(result['noise_inds_n']), 2)
          assert result['returns_n2'].dtype == np.float32

          result_num_eps = result['lengths_n2'].size
          result_num_timesteps = result['lengths_n2'].sum()
          #episodes_so_far += result_num_eps
          #timesteps_so_far += result_num_timesteps

          curr_task_results.append(result)
          # Update ob stats.

          print('ob_count', result['ob_count'])
          if result['ob_count'] > 0:
              if ob_sum is None:
                  ob_sum = np.copy(result['ob_sum'])
              else:
                  ob_sum += result['ob_sum']

              if ob_sumsq is None:
                  ob_sumsq = np.copy(result['ob_sumsq'])
              else:
                  ob_sumsq += result['ob_sumsq']

              if ob_count is None:
                  ob_count = result['ob_count']
              else:
                  ob_count += result['ob_count']
          if "timing" in result:
            worker_timing.append(_process_subworker_timing(result["timing"]))
        timing.update(_average_dicts(worker_timing))


        timing["process_obstats"] = time.time()
        #   if policy.needs_ob_stat and result['ob_count'] > 0:
        #     ob_stat.increment(result['ob_sum'], result['ob_sumsq'], result['ob_count'])
        #     ob_count_this_batch += result['ob_count']
        #
        # print("NOISELESS RETURNS:", np.mean(test_returns))
        # print("NOISELESS LENGTHS:", np.mean(test_lengths))

        # Assemble the results.
        noise_inds_n = np.concatenate([r['noise_inds_n'] for
                                       r in curr_task_results])
        returns_n2 = np.concatenate([r['returns_n2'] for r in curr_task_results])
        lengths_n2 = np.concatenate([r['lengths_n2'] for r in curr_task_results])
        assert noise_inds_n.shape[0] == returns_n2.shape[0] == lengths_n2.shape[0]
        # Process the returns.
        if config.return_proc_mode == "centered_rank":
          proc_returns_n2 = utils.compute_centered_ranks(returns_n2)
        else:
          raise NotImplementedError(config.return_proc_mode)
        timing["process_returns"] = time.time()
        # Compute and take a step
        g, count = utils.batched_weighted_sum(
            proc_returns_n2[:, 0] - proc_returns_n2[:, 1],
            (noise.get(idx, policy_num_params) for idx in noise_inds_n),
            batch_size=500)
        #g /= returns_n2.size
        assert (g.shape == (policy_num_params,) and g.dtype == np.float32 and
                count == len(noise_inds_n))
        timing["end"] = time.time()

        if len(test_returns) == 0:
            return (0,
                    0,
                    0,
                    g,
                    returns_n2.size,
                    (ob_sum, ob_sumsq, ob_count), timing)

        return (len(test_returns),
                np.mean(test_returns),
                np.mean(test_lengths),
                g,
                returns_n2.size,
                (ob_sum, ob_sumsq, ob_count), timing)


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
  parser.add_argument("--num-episodes", default=1, type=int,
                      help="The approximate number of episodes per update.")
  parser.add_argument("--num-timesteps", default=0, type=int,
                      help="The approximate number of episodes per update.")
  parser.add_argument("--min-task-runtime", default=0.2, type=float,
                      help="The minimum time per batch.")
  parser.add_argument("--warmup", default=0, type=int,
                      help="Warm up the plasma manager connections.")
  parser.add_argument("--num-master-workers", default=1, type=int,
                      help="Warm up the plasma manager connections.")
  parser.add_argument("--num-workers-per-master", default=1, type=int,
                      help="Warm up the plasma manager connections.")
  parser.add_argument("--num-episode-pairs-per-worker", default=1, type=int,
                      help="Warm up the plasma manager connections.")
  parser.add_argument("--pin-workers", action='store_true',
                      help="pin the actors to specific cores.")




  args = parser.parse_args()
  num_workers = args.num_workers
  env_name = args.env_name
  stepsize = args.stepsize

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
    ids = [f.remote() for _ in range(201)]

    i = 0
    for obj_id in ids:
      print(i)
      ray.get([g.remote(obj_id) for _ in range(201)])
      i += 1

    print("Finished running a bunch of tasks")
  else:
    print("Not warming up the object manager connections")

  config = Config(l2coeff=0.005,
                  noise_stdev=0.02,
                  episodes_per_batch=args.num_episodes,
                  #time_per_batch=args.time_per_batch,
                  timesteps_per_batch=args.num_timesteps,
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
  print("Creating master actors.")

  master_actors = [MasterWorker.remote(args.num_workers_per_master, config, policy_params, env_name, [noise_id],
                                       min_task_runtime=args.min_task_runtime,
                                       num_episode_pairs_per_worker=args.num_episode_pairs_per_worker,
                                       pin_workers=args.pin_workers)
                   for _ in range(args.num_master_workers)]

  print("waiting for master actors to finish starting")

  ray.get([ma.no_op.remote() for ma in master_actors])

  print("master actors have started")

  # check workers per machine
  r = ray.worker.global_worker.redis_client
  max_num_workers = np.max([int(r.get(key).decode('ascii')) for key in r.keys("pin*")])
  print("Some machine has ",
        max_num_workers,
        " workers on it.")

  if max_num_workers > 32:
      raise Exception("Too many workers on the same machine")


  env = gym.make(env_name)
  sess = utils.make_session(single_threaded=False)
  policy = policies.MujocoPolicy(env.observation_space, env.action_space,
                                 **policy_params)
  tf_util.initialize()
  optimizer = optimizers.Adam(policy, stepsize)

  ob_stat = utils.RunningStat(env.observation_space.shape, eps=1e-2)

  if args.warmup == 1:
    print("Sleeping for 5s")
    time.sleep(5)
    print("Finished sleeping")

  episodes_so_far = 0
  timesteps_so_far = 0
  tstart = time.time()

  iteration = 0

  result_info = []

  from collections import OrderedDict
  from csv import DictWriter
  timing = OrderedDict()
  filename = "es_{}_{}_{}_{}.pickle".format(args.num_workers, args.test_prob, args.num_episodes, time.time())
  resultfile = open(filename, 'w')
  writer = DictWriter(resultfile, ["start", "put"])
  writer.writeheader()

  for _ in range(100):
    theta = policy.get_trainable_flat()
    assert theta.dtype == np.float32

    # Put the current policy weights in the object store.

    timing["start"] = time.time()
    theta_id = ray.put(theta)
    timing["put"] = time.time()

    # Divide by 2 because each one does 2.
    #num_to_wait_for = int(np.ceil(args.num_episodes / 2))
    #num_batches = int(np.ceil(args.num_episodes / 2 / len(workers)))

    # Use the actors to do rollouts, note that we pass in the ID of the policy
    # weights.
    rollout_ids = []

    results_and_grad_ids = [ma.do_rollouts_and_return_update.remote(
            [theta_id], policy.num_params,
            ob_stat.mean if policy.needs_ob_stat else None,
            ob_stat.std if policy.needs_ob_stat else None, time.time()) for ma in master_actors]
    timing["launch"] = time.time()

    results_and_grads = ray.get(results_and_grad_ids)
    timing["gather"] = time.time()

    total_noiseless_score = 0
    total_noiseless_length = 0
    total_num_noiseless = 0
    total_grad = np.zeros_like(theta)
    total_returns = 0
    worker_timings = []

    for num_noiseless, noiseless_score, noiseless_length, grad, num_returns, obstat_info, ma_timing in results_and_grads:
        total_noiseless_score += noiseless_score * num_noiseless
        total_noiseless_length += noiseless_length * num_noiseless
        total_num_noiseless += num_noiseless

        total_grad += grad
        total_returns += num_returns

        if policy.needs_ob_stat and obstat_info[0] is not None:
            ob_stat.increment(*obstat_info)
            #ob_count_this_batch += result['ob_count']
        worker_timings.append(_process_ma_timing(ma_timing))
    import ipdb; ipdb.set_trace()
    avg_results = _average_dicts(worker_timings)

    total_grad /= total_returns
    update_ratio = optimizer.update(-total_grad + config.l2coeff * theta)
    total_noiseless_score /= total_num_noiseless
    total_noiseless_length /= total_num_noiseless
    print("NOISELESS SCORE: ", total_noiseless_score)
    print("NOISELESS LENGTH: ", total_noiseless_length)
    timing["update"] = time.time()

    # Update ob stat (we're never running the policy in the master, but we
    # might be snapshotting the policy).
    if policy.needs_ob_stat:
      policy.set_ob_stat(ob_stat.mean, ob_stat.std)

    step_tend = time.time()

    iteration += 1
    print("iteration ", iteration)
    print("total time elapsed ", time.time() - timing["start"])
    writer.writerow(dict(timing))

    # if total_noiseless_score >= 6000:
    #   filename = "es_{}_{}_{}_{}.pickle".format(args.num_workers, args.test_prob, args.num_episodes, time.time())
    #   print("\n\nBreaking and storing results in ", filename, "\n\n")
    #   break

import time
results_file = open(filename, 'wb')
pickle.dump(result_info, results_file)
