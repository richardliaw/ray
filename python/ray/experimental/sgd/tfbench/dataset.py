from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod
import os
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import cPickle
import tensorflow as tf
from tensorflow.python.platform import gfile

from ray.experimental.sgd.tfbench import model_config, preprocessing
from ray.experimental.sgd.model import Model
from ray.experimental.tfutils import TensorFlowVariables

_SUPPORTED_INPUT_PREPROCESSORS = {
    'imagenet': {
        'default': preprocessing.RecordInputImagePreprocessor,
        'official_models_imagenet': preprocessing.ImagenetPreprocessor,
    },
    'cifar10': {
        'default': preprocessing.Cifar10ImagePreprocessor
    },
}


class Dataset(object):
    """Abstract class for cnn benchmarks dataset."""

    def __init__(self,
                 name,
                 data_dir=None,
                 queue_runner_required=False,
                 num_classes=None):
        self.name = name
        self.data_dir = data_dir
        self._queue_runner_required = queue_runner_required
        self._num_classes = num_classes

    def tf_record_pattern(self, subset):
        return os.path.join(self.data_dir, '%s-*-of-*' % subset)

    def reader(self):
        return tf.TFRecordReader()

    @property
    def num_classes(self):
        return self._num_classes

    @num_classes.setter
    def num_classes(self, val):
        self._num_classes = val

    @abstractmethod
    def num_examples_per_epoch(self, subset):
        pass

    def __str__(self):
        return self.name

    def get_input_preprocessor(self, input_preprocessor='default'):
        assert not self.use_synthetic_gpu_inputs()
        return _SUPPORTED_INPUT_PREPROCESSORS[self.name][input_preprocessor]

    def queue_runner_required(self):
        return self._queue_runner_required

    def use_synthetic_gpu_inputs(self):
        return not self.data_dir


class ImageDataset(Dataset):
    """Abstract class for image datasets."""

    def __init__(self,
                 name,
                 height,
                 width,
                 depth=None,
                 data_dir=None,
                 queue_runner_required=False,
                 num_classes=1001):
        super(ImageDataset, self).__init__(name, data_dir,
                                           queue_runner_required, num_classes)
        self.height = height
        self.width = width
        self.depth = depth or 3


class Cifar10Dataset(ImageDataset):
    """Configuration for cifar 10 dataset.
    It will mount all the input images to memory.
    """

    def __init__(self, data_dir=None):
        super(Cifar10Dataset, self).__init__(
            'cifar10',
            32,
            32,
            data_dir=data_dir,
            queue_runner_required=True,
            num_classes=11)

    def read_data_files(self, subset='train'):
        """Reads from data file and returns images and labels in a numpy array."""
        assert self.data_dir, (
            'Cannot call `read_data_files` when using synthetic '
            'data')
        if subset == 'train':
            filenames = [
                os.path.join(self.data_dir, 'data_batch_%d' % i)
                for i in xrange(1, 6)
            ]
        elif subset == 'validation':
            filenames = [os.path.join(self.data_dir, 'test_batch')]
        else:
            raise ValueError('Invalid data subset "%s"' % subset)

        inputs = []
        for filename in filenames:
            with gfile.Open(filename, 'rb') as f:
                # python2 does not have the encoding parameter
                encoding = {} if six.PY2 else {'encoding': 'bytes'}
                inputs.append(cPickle.load(f, **encoding))
        # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
        # input format.
        all_images = np.concatenate(
            [each_input[b'data'] for each_input in inputs]).astype(np.float32)
        all_labels = np.concatenate(
            [each_input[b'labels'] for each_input in inputs])
        return all_images, all_labels

    def num_examples_per_epoch(self, subset='train'):
        if subset == 'train':
            return 50000
        elif subset == 'validation':
            return 10000
        else:
            raise ValueError('Invalid data subset "%s"' % subset)


if __name__ == '__main__':
    data = Cifar10Dataset()
