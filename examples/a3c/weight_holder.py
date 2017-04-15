class WeightHolder():
    def __init__(self, policy):
        self.variables = policy.variables
        self.var_list = policy.var_list

    def get_weights(self):
        if not hasattr(self, "_weights"):
            self._weights = self.variables.get_weights()
        return self._weights

    def set_weights(self, weights):
        self._weights = weights

    def model_update(self, grads, step):
        for var, grad in grads.items():
            print(grad)
            self._weights[var] -= step * grad
