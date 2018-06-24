from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import defaultdict
import os
import tensorflow as tf

import ray
from ray.rllib.optimizers.policy_evaluator import TFMultiGPUSupport
from ray.rllib.optimizers.policy_optimizer import PolicyOptimizer
from ray.rllib.optimizers.sample_batch import SampleBatch
from ray.rllib.optimizers.multi_gpu_impl import LocalSyncParallelOptimizer
from ray.rllib.utils.timer import TimerStat


class LocalMultiGPUOptimizer(PolicyOptimizer):
    """A synchronous optimizer that uses multiple local GPUs.

    Samples are pulled synchronously from multiple remote evaluators,
    concatenated, and then split across the memory of multiple local GPUs.
    A number of SGD passes are then taken over the in-memory data. For more
    details, see `multi_gpu_impl.LocalSyncParallelOptimizer`.

    This optimizer is Tensorflow-specific and require evaluators to implement
    the TFMultiGPUSupport API.
    """

    def _init(self, sgd_batch_size=128, sgd_stepsize=5e-5, num_sgd_iter=10,
              timesteps_per_batch=1024):
        self.batch_size = sgd_batch_size
        self.sgd_stepsize = sgd_stepsize
        self.num_sgd_iter = num_sgd_iter
        self.timesteps_per_batch = timesteps_per_batch
        gpu_ids = ray.get_gpu_ids()
        if not gpu_ids:
            self.devices = ["/cpu:0"]
        else:
            self.devices = ["/gpu:{}".format(i) for i in range(len(gpu_ids))]
        self.batch_size = int(
                sgd_batch_size / len(self.devices)) * len(self.devices)
        assert self.batch_size % len(self.devices) == 0
        assert self.batch_size >= len(self.devices), "batch size too small"
        self.per_device_batch_size = int(self.batch_size / len(self.devices))
        self.sample_timer = TimerStat()
        self.load_timer = TimerStat()
        self.grad_timer = TimerStat()
        self.update_weights_timer = TimerStat()

        print("LocalMultiGPUOptimizer devices", self.devices)
        print("LocalMultiGPUOptimizer batch size", self.batch_size)

        # List of (feature name, feature placeholder) tuples
        self.loss_inputs = self.local_evaluator.for_policy(lambda pi: pi.loss_in)

        # per-GPU graph copies created below must share vars with the policy
        # reuse is set to AUTO_REUSE because Adam nodes are created after
        # all of the device copies are created.
        with self.local_evaluator.sess.graph.as_default():
            main_thread_scope = tf.get_variable_scope()
            with tf.variable_scope(main_thread_scope, reuse=tf.AUTO_REUSE):
                def build_loss(inputs):
                    cfg = self.local_evaluator.policy_config
                    ac_space = self.local_evaluator.env.get_unwrapped().action_space
                    return self.local_evaluator.policy_graph(None, ac_space, cfg, inputs)

                self.par_opt = LocalSyncParallelOptimizer(
                    tf.train.AdamOptimizer(self.sgd_stepsize),
                    self.devices,
                    self.loss_inputs,
                    self.per_device_batch_size,
                    build_loss,
                    os.getcwd())

            self.sess = self.local_evaluator.sess
            self.sess.run(tf.global_variables_initializer())

    def step(self, postprocess_fn=None):
        with self.update_weights_timer:
            if self.remote_evaluators:
                weights = ray.put(self.local_evaluator.get_weights())
                for e in self.remote_evaluators:
                    e.set_weights.remote(weights)

        with self.sample_timer:
            if self.remote_evaluators:
                # TODO(rliaw): remove when refactoring
                from ray.rllib.ppo.rollout import collect_samples
                samples = collect_samples(self.remote_evaluators,
                                          self.timesteps_per_batch)
            else:
                samples = self.local_evaluator.sample()
            assert isinstance(samples, SampleBatch)

            if postprocess_fn:
                postprocess_fn(samples)

        with self.load_timer:
            tuples_per_device = self.par_opt.load_data(
                self.local_evaluator.sess,
                samples.columns([key for key, _ in self.loss_inputs]))

        with self.grad_timer:
            all_extra_fetches = defaultdict(list)
            ev = self.local_evaluator
            num_batches = (
                int(tuples_per_device) // int(self.per_device_batch_size))
            for i in range(self.num_sgd_iter):
                iter_extra_fetches = defaultdict(list)
                permutation = np.random.permutation(num_batches)
                for batch_index in range(num_batches):
                    # TODO(ekl) support ppo's debugging features, e.g.
                    # printing the current loss and tracing
                    batch_fetches = self.par_opt.optimize(
                        self.sess,
                        permutation[batch_index] * self.per_device_batch_size,
                        extra_feed_dict=ev.for_policy(lambda pi: pi.extra_apply_grad_feed_dict()))
                    for k, v in batch_fetches.items():
                        iter_extra_fetches[k] += [v]
                for k, v in iter_extra_fetches.items():
                    all_extra_fetches[k] += [v]

        self.num_steps_sampled += samples.count
        self.num_steps_trained += samples.count
        return all_extra_fetches

    def stats(self):
        return dict(PolicyOptimizer.stats(), **{
            "sample_time_ms": round(1000 * self.sample_timer.mean, 3),
            "load_time_ms": round(1000 * self.load_timer.mean, 3),
            "grad_time_ms": round(1000 * self.grad_timer.mean, 3),
            "update_time_ms": round(1000 * self.update_weights_timer.mean, 3),
        })
