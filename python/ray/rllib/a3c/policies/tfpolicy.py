from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import ray
import gym
from ray.rllib.a3c.policy import Policy

class TFPolicy(Policy):
    """The policy base class for TensorFlow Policies."""
    def __init__(self, ob_space, action_space,
                 model_cls, dist_type=None, loss=None, config={}):
        self.local_steps = 0
        self.action_space = action_space
        self.summarize = summarize
        worker_device = "/job:localhost/replica:0/task:0/cpu:0"
        self.g = tf.Graph()
        with self.g.as_default(), tf.device(worker_device):
            with tf.variable_scope(name):
                self.setup_graph(ob_space, action_space)
                assert all([hasattr(self, attr)
                            for attr in ["vf", "logits", "x", "var_list"]])
            print("Setting up loss")

            if loss:
                self._setup_loss(loss)
            self._initialize()

    def _initialize(self):
        self.sess = tf.Session(graph=self.g, config=tf.ConfigProto(
            intra_op_parallelism_threads=1, inter_op_parallelism_threads=2))
        self.sess.run(tf.global_variables_initializer())

    def compute_action(self, observations):
        raise NotImplementedError

    def value(self, ob):
        raise NotImplementedError
