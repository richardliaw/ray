from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import distutils.version
import ray
from policy import *

class CNNPolicy(Policy):

    def setup_graph(self, ob_space, ac_space):
        self.x = tf.placeholder(tf.float32,
                shape=[None] + list(ob_space)) # TODO!

        conv1 = tf.contrib.layers.convolution2d(
                inputs=self.x,
                num_outputs=16,
                kernel_size=(8, 8),
                stride=4,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer())
        conv2 = tf.contrib.layers.convolution2d(
                inputs=conv1,
                num_outputs=32,
                kernel_size=(4, 4),
                stride=2,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer())
        fc1 = tf.contrib.layers.fully_connected(
                inputs=tf.contrib.layers.flatten(conv2),
                num_outputs=256,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer())

        self.logits = tf.contrib.layers.fully_connected(
                inputs=fc1,
                num_outputs=ac_space,
                activation_fn=None)

        self.vf = tf.contrib.layers.fully_connected(
                inputs=fc1,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer())

        # op to sample an action - multinomial takes unnormalized log probs
        # self.logits = tf.Print(self.logits, [self.logits], "self.Logits - ")
        self.sample = tf.reshape(tf.multinomial(self.logits, 1), []) #TODO: change to categorical?
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
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

    def act(self, ob, c, h):
        return self.sess.run([self.sample, self.vf, None],
                        {self.x: [ob]})

    def value(self, ob, c, h):
        return self.sess.run(self.vf, {self.x: [ob]})[0]

    def get_initial_features(self):
        return None
