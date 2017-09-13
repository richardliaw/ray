from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from ray.rllib.models.misc import linear, normc_initializer
from ray.rllib.a3c.policy import Policy
from ray.rllib.models.catalog import ModelCatalog


class SharedModel(Policy):
    def __init__(self, ob_space, ac_space, config, **kwargs):
        super(SharedModel, self).__init__(ob_space, ac_space, config, **kwargs)

    def setup_graph(self, ob_space, ac_space):
        self.x = tf.placeholder(tf.float32, [None] + list(ob_space))
        dist_class, self.logit_dim = ModelCatalog.get_action_dist(ac_space)
        self._model = ModelCatalog.get_model(self.x, self.logit_dim)
        self.logits = self._model.outputs
        self.curr_dist = dist_class(self.logits)
        # with tf.variable_scope("vf"):
        #     vf_model = ModelCatalog.get_model(self.x, 1)
        self.vf = tf.reshape(linear(self._model.last_layer, 1, "value",
                                    normc_initializer(1.0)), [-1])

        self.sample = self.curr_dist.sample()
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          tf.get_variable_scope().name)
        self.global_step = tf.get_variable(
            "global_step", [], tf.int32,
            initializer=tf.constant_initializer(0, dtype=tf.int32),
            trainable=False)

    def get_gradients(self, batch):
        info = {}
        feed_dict = {
            self.x: batch.si,
            self.ac: batch.a,
            self.adv: batch.adv,
            self.r: batch.r,
        }
        self.grads = [g for g in self.grads if g is not None]
        self.local_steps += 1
        if self.summarize:
            grad, summ = self.sess.run([self.grads, self.summary_op],
                                       feed_dict=feed_dict)
            info['summary'] = summ
        else:
            grad = self.sess.run(self.grads, feed_dict=feed_dict)
        return grad, info

    def compute_actions(self, ob, *args):
        action, vf = self.sess.run([self.sample, self.vf],
                                   {self.x: [ob]})
        return action[0], vf

    def value(self, ob, *args):
        return self.sess.run(self.vf, {self.x: [ob]})[0]

    def get_initial_features(self):
        return []
