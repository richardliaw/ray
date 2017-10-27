class Model(object):
    def get_weights(self):
        raise NotImplementedError

    def set_weights(self, weights):
        raise NotImplementedError

    def get_initial_features(self):
        raise NotImplementedError


class TFModel(Model):
    """TODO(rliaw): Probably want to start session here"""
    is_recurrent = None

    def __init__(self, config):
        self.sess = None

    def _setup_graph(self):
        raise NotImplementedError

    def _set_tf_vars(heads):
        self.weights = ray.experimental.TensorFlowVariables(
            [self.logits, self.vf],
            self.sess)

    def get_weights(self):
        weights = self.weights.get_weights()
        return weights

    def set_weights(self, weights):
        self.weights.set_weights(weights)

    def get_initial_features(self):
        raise NotImplementedError
