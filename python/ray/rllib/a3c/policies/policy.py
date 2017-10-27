from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import ray
import gym


class Policy(object):
    """The policy base class.
    This terminology is not exactly correct, as it encompasses both the
    actor and the critic, while typically the actor is the policy."""

    def __init__(self, ob_space, action_space, model_cls, dist_type=None, config={}):
        raise NotImplementedError

    def compute_action(self, ob):
        """Compute action for a _single_ observation"""
        raise NotImplementedError

    def value(self, ob):
        """Compute value for a _single_ observation"""
        raise NotImplementedError
