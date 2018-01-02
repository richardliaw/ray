# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class GDOptimizer(object):
    def __init__(self, weights):
        self.weights = weights
        self.dim = weights.shape[0]
        self.t = 0

    def update(self, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        self.weights += step
        # ratio = np.linalg.norm(step) / np.linalg.norm(theta)
        # return self.weights

    def _compute_step(self, globalg):
        raise NotImplementedError


class SGD(GDOptimizer):
    def __init__(self, weights, stepsize, momentum=0.9):
        GDOptimizer.__init__(self, weights)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.stepsize, self.momentum = stepsize, momentum

    def _compute_step(self, globalg):
        self.v = self.momentum * self.v + (1. - self.momentum) * globalg
        step = -self.stepsize * self.v
        return step


class Adam(GDOptimizer):
    def __init__(self, weights, stepsize, beta1=0.9, beta2=0.999, epsilon=1e-08):
        GDOptimizer.__init__(self, weights)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, globalg):
        a = self.stepsize * (np.sqrt(1 - self.beta2 ** self.t) /
                             (1 - self.beta1 ** self.t))
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step
