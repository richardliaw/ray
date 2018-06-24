from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import tensorflow as tf

import ray
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils.process_rollout import compute_advantages
from ray.rllib.utils.tf_policy_graph import TFPolicyGraph

# Name scope in which created variables will be placed under
TOWER_SCOPE_NAME = "tower"



class TFMultiGPUGraph(TFPolicyGraph):

    def __init__(self, *args):
        self._graph_args = args
        self._get_devices()
        self._graph_cls = graph_cls

    def __call__(self, *args):
        main_thread_scope = tf.get_variable_scope()
        with tf.variable_scope(main_thread_scope, reuse=tf.AUTO_REUSE):
            shared_graph = self._graph_cls(*self._graph_args)
            input_placeholders = shared_graph.loss_in

            # Split on the CPU in case the data doesn't fit in GPU memory.
            with tf.device("/cpu:0"):
                names, placeholders = zip(*input_placeholders)
                data_splits = zip(
                    *[tf.split(ph, len(devices)) for ph in placeholders])

            towers = []
            for device, device_placeholders in zip(self.devices, data_splits):
                towers.append(
                    self._setup_device(args, device, zip(names, device_placeholders)))

            avg = average_gradients([t.grads for t in towers])
            if grad_norm_clipping:
                for i, (grad, var) in enumerate(avg):
                    if grad is not None:
                        avg[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)
            shared_graph.optimizer = lambda: tf.train.GradientDescent()
            shared_graph.gradients = lambda x: avg
            shared
            shared_graph.initialize()
            return shared_graph

    def _setup_device(self, args, device, device_input_placeholders):
        with tf.device(device):
            with tf.name_scope(TOWER_SCOPE_NAME):
                device_input_batches = []
                device_input_slices = []
                for name, ph in device_input_placeholders:
                    current_batch = tf.Variable(
                        ph, trainable=False, validate_shape=False,
                        collections=[])
                    device_input_batches.append(current_batch)
                    current_slice = tf.slice(
                        current_batch,
                        [self._batch_index] + [0] * len(ph.shape[1:]),
                        ([self.per_device_batch_size] + [-1] *
                         len(ph.shape[1:])))
                    current_slice.set_shape(ph.shape)
                    device_input_slices.append((name, current_slice))
                graph = self._graph_cls(*args, loss_in=device_input_slices)
                device_grads = graph.gradients(self.optimizer)
            return Tower(
                tf.group(*[batch.initializer
                           for batch in device_input_batches]),
                device_grads,
                graph)

    def _get_devices(self):
        gpu_ids = ray.get_gpu_ids()
        if not gpu_ids:
            self.devices = ["/cpu:0"]
        else:
            self.devices = ["/gpu:{}".format(i) for i in range(len(gpu_ids))]

graph = PolicyGraphCls
new_graph = TFMultiGPUBuilder(graph)
policy_graph = new_graph(obs_space, ac_space, config)
