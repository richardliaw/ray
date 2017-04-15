class WeightHolder():
	def __init__(self, policy):
		self.variables = policy.variables

    def get_weights(self):
        if not hasattr(self, "_weights"):
            self._weights = self.variables.get_weights()
        return self._weights

    def set_weights(self, weights):
        self._weights = weights

    def model_update(self, grads, step):
        for var, grad in zip(self.var_list, grads):
            self._weights[var.name[:-2]] -= step * grad