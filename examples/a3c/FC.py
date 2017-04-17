from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import distutils.version
import ray
from policy import *

class FCPolicy(Policy):

    def setup_graph(self, ob_space, ac_space):
        self.x = tf.placeholder(tf.float32,
                shape=[None] + list(ob_space)) 

        # fc0 = tf.contrib.layers.fully_connected(
        #         inputs=self.x,
        #         num_outputs=128,
        #         activation_fn=tf.nn.elu,
        #         weights_initializer=normalized_columns_initializer(0.1),
        #         scope="fc0")

        fc1 = tf.contrib.layers.fully_connected(
                inputs=self.x,
                num_outputs=128,
                activation_fn=tf.nn.elu,
                weights_initializer=normalized_columns_initializer(0.1),
                scope="fc1")

        fc2 = tf.contrib.layers.fully_connected(
                inputs=fc1,
                num_outputs=128,
                activation_fn=tf.nn.elu,
                weights_initializer=normalized_columns_initializer(0.1),
                scope="fc2")

        self.logits = linear(fc2, ac_space, "action", normalized_columns_initializer(0.001))

        self._vf = tf.contrib.layers.fully_connected(
                inputs=fc2,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(0.001),
                scope="vf")
        self.vf = tf.reshape(self._vf, [-1])

        # op to sample an action - multinomial takes unnormalized log probs
        self.sample = categorical_sample(self.logits, ac_space)[0, :]

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        self.feature = [tf.no_op()]
        self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)

    def get_gradients(self, batch):
        """Computing the gradient is actually model-dependent."""
        # TODO: Fix batch object here
        feed_dict = {
            self.x: batch.si,
            self.ac: batch.a,
            self.adv: batch.adv,
            self.r: batch.r,
        }
        self.local_steps += 1
        return self.sess.run(self.grads, feed_dict=feed_dict)

    def act(self, ob, *args):
        return self.sess.run([self.sample, self.vf, ] + self.feature,
                        {self.x: [ob]})

    def value(self, ob, *args):
        return self.sess.run(self.vf, {self.x: [ob]})[0]

    def get_initial_features(self):
        return [None]
