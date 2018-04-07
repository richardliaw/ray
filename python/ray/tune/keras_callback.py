from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras


class TuneCallback(keras.callbacks.Callback):
    """A callback to integrate Ray Tune with Keras"""

    def __init__(self, reporter, logs={}):
        self.reporter = reporter
        self.iteration = 0

    def on_train_end(self, epoch, logs={}):
        self.reporter(
            timesteps_total=self.iteration, done=1, mean_accuracy=logs["acc"])

    def on_batch_end(self, batch, logs={}):
        self.iteration += 1
        self.reporter(timesteps_total=self.iteration, mean_accuracy=logs["acc"])
