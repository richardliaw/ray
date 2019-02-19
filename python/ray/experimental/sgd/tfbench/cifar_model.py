from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from ray.experimental.sgd.tfbench import model_config, dataset
from ray.experimental.sgd.model import Model
from ray.experimental.tfutils import TensorFlowVariables


def next_batch(num, data, labels):
    """Return a total of `num` random samples and labels."""
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = data[idx]
    labels_shuffle = labels[idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


class CIFARModel(Model):
    def __init__(self, batch=128, use_cpus=False):
        self.batch = batch

        self.data = dataset.Cifar10Dataset(
            data_dir='/tmp/cifar10_data/cifar-10-batches-py')
        self._model = model_config.get_model_config("resnet20", self.data)

        self._all_x, self._all_y = self.data.read_data_files()
        self._all_x = self._all_x.reshape(
            [len(self._all_x)] + self._model.get_input_shapes()[0][1:])

        self.inputs = tf.placeholder(
            tf.float32, self._model.get_input_shapes()[0], name="x")
        self.labels = tf.placeholder(
            tf.int64, self._model.get_input_shapes()[1], name="y_")

        logits, aux = self._model.build_network(
            self.inputs, data_format=use_cpus and "NHWC" or "NCHW")
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=self.labels)

        # Implement model interface
        self.loss = tf.reduce_mean(loss, name='xentropy-loss')
        self.optimizer = tf.train.GradientDescentOptimizer(1e-6)

        self.variables = TensorFlowVariables(self.loss,
                                             tf.get_default_session())

    def get_loss(self):
        return self.loss

    def get_optimizer(self):
        return self.optimizer

    def get_feed_dict(self):
        mini_x, mini_y = next_batch(self.batch, self._all_x, self._all_y)
        return {self.inputs: mini_x, self.labels: mini_y}

    def get_weights(self):
        return self.variables.get_flat()

    def set_weights(self, weights):
        self.variables.set_flat(weights)


if __name__ == '__main__':
    session = tf.Session()
    model = CIFARModel(use_cpus=True)
    fd = model.get_feed_dict()
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    session.run(init_op)
    g = session.run(model.loss, feed_dict=fd)
    print(g)
