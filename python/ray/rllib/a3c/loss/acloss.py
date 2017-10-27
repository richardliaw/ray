from loss import TFLoss
import tensorflow as tf


class ACLoss(TFLoss):

    def _init(self, policy):
        super(self, ACLoss)._init(policy)

        self.var_list = policy.get_var_list()  # TF specific
        self.ac = self._get_action_placeholder(policy.action_space)
        self.adv = tf.placeholder(tf.float32, [None], name="adv")
        self.r = tf.placeholder(tf.float32, [None], name="r")

        log_prob = self.policy.logp(self.ac)

        # The "policy gradients" loss: its derivative is precisely the policy
        # gradient. Notice that self.ac is a placeholder that is provided
        # externally. adv will contain the advantages, as calculated in
        # process_rollout.
        self.pi_loss = - tf.reduce_sum(log_prob * self.adv)

        delta = self.vf - self.r
        self.vf_loss = 0.5 * tf.reduce_sum(tf.square(delta))
        self.entropy = tf.reduce_sum(self.policy.entropy())
        self.loss = self.pi_loss + 0.5 * self.vf_loss - self.entropy * 0.01
        self._setup_optimizer()

    def _get_action_placeholder(self, action_space):
        if isinstance(action_space, gym.spaces.Box):
            ac_size = action_space.shape[0]
            ac = tf.placeholder(tf.float32, [None, ac_size], name="ac")
        elif isinstance(action_space, gym.spaces.Discrete):
            ac = tf.placeholder(tf.int64, [None], name="ac")
        else:
            raise NotImplemented(
                "action space" + str(type(action_space)) +
                "currently not supported")
        return ac

    def _setup_optimizer(self):
        grads = tf.gradients(self.loss, self.var_list)
        self.grads, _ = tf.clip_by_global_norm(grads, 40.0)
        grads_and_vars = list(zip(self.grads, self.var_list))
        self._opt = tf.train.AdamOptimizer(1e-4)
        self._apply_gradients = self._opt.apply_gradients(grads_and_vars)


    def compute_gradients(self, batch):
        """Computing the gradient is actually model-dependent.
        """
        feed_dict = {
            self.x: batch.si,
            self.ac: batch.a,
            self.adv: batch.adv,
            self.r: batch.r,
            self.state_in[0]: batch.features[0],
            self.state_in[1]: batch.features[1]
        }
        info = {}
        grad = self._sess.run(self.grads, feed_dict=feed_dict)
        return grad, info

    def apply_gradients(self, grads):
        feed_dict = {self.grads[i]: grads[i]
                     for i in range(len(grads))}
        self._sess.run(self._apply_gradients, feed_dict=feed_dict)
