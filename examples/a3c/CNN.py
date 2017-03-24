rom __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import distutils.version
import ray
from policy import *

class CNNPolicy(Policy):

    def setup_graph(self, hparams):
    	print [None] + (hparams['input_size'])
        self._input = tf.placeholder(tf.float32,
                shape=[None] + (hparams['input_size'])) # TODO!

        conv1 = tf.contrib.layers.convolution2d(
                inputs=self._input,
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
                num_outputs=hparams['num_actions'],
                activation_fn=None)

        self._value = tf.contrib.layers.fully_connected(
                inputs=fc1,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer())

        # op to sample an action - multinomial takes unnormalized log probs
        # self.logits = tf.Print(self.logits, [self.logits], "self.Logits - ")
        self._sample = tf.reshape(tf.multinomial(self.logits, 1), [])

    def get_gradients(self, batch):
        """Computing the gradient is actually model-dependent.
            The LSTM needs its hidden states in order to compute the gradient accurately."""
        feed_dict = {
            self.x: batch.si,
            self.ac: batch.a,
            self.adv: batch.adv,
            self.r: batch.r,
            self.state_in[0]: batch.features[0],
            self.state_in[1]: batch.features[1],
        }
        self.local_steps += 1
        return self.sess.run(self.grads, feed_dict=feed_dict)

    def act(self, ob, c, h):
        return self.sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

    def value(self, ob, c, h):
        return self.sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]

    def get_initial_features(self):
        return self.state_init
