from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils.process_rollout import compute_advantages
from ray.rllib.utils.tf_policy_graph import TFPolicyGraph


class PPOTFPolicyGraph(TFPolicyGraph):
    """PPO Graph"""

    def __init__(self, ob_space, action_space, config):
        self.config = config
        self._setup_graph(ob_space, action_space)
        print("Setting up loss")
        loss = self.setup_loss(action_space)
        self.is_training = tf.placeholder_with_default(True, ())
        self.sess = tf.get_default_session()

        TFPolicyGraph.__init__(
            self, self.sess, obs_input=self.obs,
            action_sampler=self.sampler, loss=loss,
            loss_inputs=self.loss_in, is_training=self.is_training)

    def _setup_graph(self, ob_space, ac_space):
        self.obs = tf.placeholder(
            tf.float32, shape=(None,) + ob_space.shape)

        self.dist_cls, self.logit_dim = ModelCatalog.get_action_dist(ac_space)
        self.logits = ModelCatalog.get_model(
            self.obs, self.logit_dim, self.config["model"]).outputs
        self.curr_dist = self.dist_cls(self.logits)
        self.sampler = self.curr_dist.sample()
        if self.config["use_gae"]:
            vf_config = self.config["model"].copy()
            # Do not split the last layer of the value function into
            # mean parameters and standard deviation parameters and
            # do not make the standard deviations free variables.
            vf_config["free_log_std"] = False
            with tf.variable_scope("value_function"):
                self.value_function = ModelCatalog.get_model(
                    self.obs, 1, vf_config).outputs
            self.value_function = tf.reshape(self.value_function, [-1])

    def setup_loss(self, action_space):
        # Defines the training inputs:
        # The coefficient of the KL penalty.
        self.kl_coeff = tf.placeholder(
            name="newkl", shape=(), dtype=tf.float32)

        self.actions = ModelCatalog.get_action_placeholder(action_space)
        # Targets of the value function.
        self.value_targets = tf.placeholder(tf.float32, shape=(None,))
        # Advantage values in the policy gradient estimator.
        self.advantages = tf.placeholder(tf.float32, shape=(None,))
        # Log probabilities from the policy before the policy update.
        self.prev_logits = tf.placeholder(
            tf.float32, shape=(None, self.logit_dim))
        self.prev_dist = self.dist_cls(self.prev_logits)
        # Value function predictions before the policy update.
        self.prev_vf_preds = tf.placeholder(tf.float32, shape=(None,))

        # Make loss functions.
        self.ratio = tf.exp(self.curr_dist.logp(self.actions) -
                            self.prev_dist.logp(self.actions))
        self.kl = self.prev_dist.kl(self.curr_dist)
        self.mean_kl = tf.reduce_mean(self.kl)
        self.entropy = self.curr_dist.entropy()
        self.mean_entropy = tf.reduce_mean(self.entropy)
        self.surr1 = self.ratio * self.advantages
        self.surr2 =  self.advantages * tf.clip_by_value(
            self.ratio, 1 - self.config["clip_param"],
            1 + self.config["clip_param"])
        self.surr = tf.minimum(self.surr1, self.surr2)
        self.mean_policy_loss = tf.reduce_mean(-self.surr)

        if self.config["use_gae"]:
            # We use a huber loss here to be more robust against outliers,
            # which seem to occur when the rollouts get longer (the variance
            # scales superlinearly with the length of the rollout)
            self.vf_loss1 = tf.square(self.value_function - self.value_targets)
            vf_clipped = self.prev_vf_preds + tf.clip_by_value(
                self.value_function - self.prev_vf_preds,
                -self.config["clip_param"], self.config["clip_param"])
            self.vf_loss2 = tf.square(vf_clipped - self.value_targets)
            self.vf_loss = tf.minimum(self.vf_loss1, self.vf_loss2)
            self.mean_vf_loss = tf.reduce_mean(self.vf_loss)
            loss = tf.reduce_mean(
                -self.surr + self.kl_coeff * self.kl +
                self.config["vf_loss_coeff"] * self.vf_loss -
                self.config["entropy_coeff"] * self.entropy)
        else:
            self.mean_vf_loss = tf.constant(0.0)
            loss = tf.reduce_mean(
                -self.surr +
                self.kl_coeff * self.kl -
                self.config["entropy_coeff"] * self.entropy)

        self.loss_in = [
            ("obs", self.obs),
            ("value_targets", self.value_targets),
            ("advantages", self.advantages),
            ("actions", self.actions),
            ("logprobs", self.prev_logits),
            ("vf_preds", self.prev_vf_preds)
        ]
        return loss

    def optimizer(self):
        return tf.train.AdamOptimizer(self.config["sgd_stepsize"])

    def extra_compute_grad_fetches(self):
        if self.summarize:
            return {"summary": self.summary_op}
        else:
            return {}

    def postprocess_trajectory(self, sample_batch, other_agent_batches=None):
        last_r = 0.0
        return compute_advantages(
            sample_batch, last_r, self.config["gamma"], self.config["lambda"])


if __name__ == '__main__':
    import gym
    from collections import defaultdict
    from ray.rllib.ppo import DEFAULT_CONFIG
    cfg = DEFAULT_CONFIG.copy()
    cfg["use_gae"] = False
    sess = tf.Session()
    with sess.as_default():
        env = gym.make("CartPole-v0")
        graph = PPOTFPolicyGraph(
            env.observation_space,
            env.action_space,
            cfg)
