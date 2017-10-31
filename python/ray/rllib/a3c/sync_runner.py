from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from ray.rllib.a3c.envs import create_and_wrap
import tensorflow as tf
import six.moves.queue as queue
from ray.rllib.a3c.runner_thread import RunnerThread, env_runner
from ray.rllib.a3c.common import process_rollout, CompletedRollout
from ray.rllib.a3c.tfpolicy import TFPolicy
from ray.rllib.a3c.runner import Runner
import ray
import os


class SyncRunner(Runner):
    """Actor object to start running simulation on workers.

    The gradient computation is also executed from this object.
    """
    def __init__(self, env_creator, policy_cls, actor_id, batch_size,
                 preprocess_config, logdir):
        env = create_and_wrap(env_creator, preprocess_config)
        self.id = actor_id
        # TODO(rliaw): should change this to be just env.observation_space
        self.policy = policy_cls(env.observation_space.shape, env.action_space)
        self.runner = SyncRunnerThread(env, self.policy, batch_size)
        self.env = env
        self.logdir = logdir
        self.start()

    def start(self):
        summary_writer = tf.summary.FileWriter(
            os.path.join(self.logdir, "agent_%d" % self.id))
        self.summary_writer = summary_writer
        self.runner.start_sync(self.policy.sess, summary_writer)

    def set_sample(self, params):
        self.policy.set_weights(params)
        self.runner.sync_run()
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)
        return {"si": batch.si.copy(),
                "a": batch.a.copy(),
                "adv": np.hstack(batch.adv),
                "r": batch.r.copy(),
                "terminal": batch.terminal,
                "features": batch.features}

    def compute_gradient(self, params):
        self.policy.set_weights(params)
        self.runner.sync_run()
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)
        gradient, info = self.policy.compute_gradients(batch)
        if "summary" in info:
            self.summary_writer.add_summary(
                tf.Summary.FromString(info['summary']),
                self.policy.local_steps)
            self.summary_writer.flush()
        info = {"id": self.id,
                "size": len(batch.a)}
        return gradient, info


class SyncRunnerThread():
    """This interacts with the environment and tells it what to do."""

    def __init__(self, env, policy, num_local_steps, visualise=False):
        self.queue = queue.Queue(5)
        self.metrics_queue = queue.Queue()
        self.num_local_steps = num_local_steps
        self.env = env
        self.last_features = None
        self.policy = policy
        self.daemon = True
        self.sess = None
        self.summary_writer = None
        self.visualise = visualise

    def start_sync(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.rollout_provider = env_runner(
            self.env, self.policy, self.num_local_steps,
            self.summary_writer, self.visualise)

    def sync_run(self):
        while True:
            # The timeout variable exists because apparently, if one worker
            # dies, the other workers won't die with it, unless the timeout is
            # set to some large number. This is an empirical observation.
            item = next(self.rollout_provider)
            if isinstance(item, CompletedRollout):
                self.metrics_queue.put(item)
            else:
                self.queue.put(item, timeout=600.0)
                return

RemoteSyncRunner = ray.remote(SyncRunner)
