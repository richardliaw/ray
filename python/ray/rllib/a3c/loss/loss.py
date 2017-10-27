class Loss(object):
    """ Contains policy improvement operators """
    def __init__(self, policy):
        self._init(policy)

    def _init(self, policy):
        raise NotImplementedError

    def compute_gradient(self, input):
        raise NotImplementedError

    def apply_gradient(self, grads):
        raise NotImplementedError


class TFLoss(Loss):
    def _init(self, policy):
        assert isinstance(policy, TFPolicy), "Using non-TF Policy in TF loss!"
        self._sess = policy.sess

