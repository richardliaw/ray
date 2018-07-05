from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import json
import numpy as np
import os
import yaml

from ray.tune.result import TrainingResult
from ray.tune.log_sync import get_syncer

try:
    import tensorflow as tf
except ImportError:
    tf = None
    print("Couldn't import TensorFlow - this disables TensorBoard logging.")


class Logger(object):
    """Logging interface for ray.tune; specialized implementations follow.

    By default, the UnifiedLogger implementation is used which logs results in
    multiple formats (TensorBoard, rllab/viskit, plain json) at once.
    """

    def __init__(self, config, logdir, upload_uri=None, verbose=True):
        self.config = config
        self.logdir = logdir
        self.uri = upload_uri
        self.verbose = verbose
        self._init()

    def _init(self):
        pass

    def on_result(self, result):
        """Given a result, appends it to the existing log."""

        raise NotImplementedError

    def close(self):
        """Releases all resources used by this logger."""

        pass

    def flush(self):
        """Flushes all disk writes to storage."""

        pass


class UnifiedLogger(Logger):
    """Unified result logger for TensorBoard, rllab/viskit, plain json.

    This class also periodically syncs output to the given upload uri."""

    def _init(self):
        self._loggers = []
        for cls in [_JsonLogger, _TFLogger, _VisKitLogger]:
            if cls is _TFLogger and tf is None:
                print("TF not installed - cannot log with {}...".format(cls))
                continue
            self._loggers.append(cls(self.config, self.logdir, self.uri))
        self._log_syncer = get_syncer(
            self.logdir, self.uri, verbose=self.verbose)

    def on_result(self, result):
        for logger in self._loggers:
            logger.on_result(result)
        self._log_syncer.set_worker_ip(result.node_ip)
        self._log_syncer.sync_if_needed()

    def close(self):
        for logger in self._loggers:
            logger.close()
        self._log_syncer.sync_now(force=True)

    def flush(self):
        for logger in self._loggers:
            logger.flush()
        self._log_syncer.sync_now(force=True)
        self._log_syncer.wait()


class NoopLogger(Logger):
    def on_result(self, result):
        pass


class _JsonLogger(Logger):
    def _init(self):
        config_out = os.path.join(self.logdir, "params.json")
        with open(config_out, "w") as f:
            json.dump(self.config, f, sort_keys=True, cls=_SafeFallbackEncoder)
        local_file = os.path.join(self.logdir, "result.json")
        self.local_out = open(local_file, "w")

    def on_result(self, result):
        json.dump(result._asdict(), self, cls=_SafeFallbackEncoder)
        self.write("\n")

    def write(self, b):
        self.local_out.write(b)
        self.local_out.flush()

    def close(self):
        self.local_out.close()


def to_tf_values(result, path):
    values = []
    for attr, value in result.items():
        if value is not None:
            if type(value) in [int, float, np.float32, np.float64, np.int32]:
                values.append(
                    tf.Summary.Value(
                        tag="/".join(path + [attr]), simple_value=value))
            elif type(value) is dict:
                values.extend(to_tf_values(value, path + [attr]))
    return values


class _TFLogger(Logger):
    def _init(self):
        self._file_writer = tf.summary.FileWriter(self.logdir)

    def on_result(self, result):
        tmp = result._asdict()
        for k in [
                "config", "pid", "timestamp", "time_total_s", "timesteps_total"
        ]:
            del tmp[k]  # not useful to tf log these
        values = to_tf_values(tmp, ["ray", "tune"])
        train_stats = tf.Summary(value=values)
        self._file_writer.add_summary(train_stats, result.timesteps_total)
        timesteps_value = to_tf_values(
            {"timesteps_total": result.timesteps_total}, ["ray", "tune"])
        timesteps_stats = tf.Summary(value=timesteps_value)
        self._file_writer.add_summary(timesteps_stats, result.training_iteration)

    def flush(self):
        self._file_writer.flush()

    def close(self):
        self._file_writer.close()


class _VisKitLogger(Logger):
    def _init(self):
        # Note that we assume params.json was already created by JsonLogger
        self._file = open(os.path.join(self.logdir, "progress.csv"), "w")
        self._csv_out = csv.DictWriter(self._file, TrainingResult._fields)
        self._csv_out.writeheader()

    def on_result(self, result):
        self._csv_out.writerow(result._asdict())

    def close(self):
        self._file.close()


class _SafeFallbackEncoder(json.JSONEncoder):
    def __init__(self, nan_str="null", **kwargs):
        super(_SafeFallbackEncoder, self).__init__(**kwargs)
        self.nan_str = nan_str

    def iterencode(self, o, _one_shot=False):
        if self.ensure_ascii:
            _encoder = json.encoder.encode_basestring_ascii
        else:
            _encoder = json.encoder.encode_basestring

        def floatstr(o, allow_nan=self.allow_nan, nan_str=self.nan_str):
            return repr(o) if not np.isnan(o) else nan_str

        _iterencode = json.encoder._make_iterencode(
            None, self.default, _encoder, self.indent, floatstr,
            self.key_separator, self.item_separator, self.sort_keys,
            self.skipkeys, _one_shot)
        return _iterencode(o, 0)

    def default(self, value):
        try:
            if np.isnan(value):
                return None
            if np.issubdtype(value, float):
                return float(value)
            if np.issubdtype(value, int):
                return int(value)
        except Exception:
            return str(value)  # give up, just stringify it (ok for logs)


def pretty_print(result):
    result = result._replace(config=None)  # drop config from pretty print
    out = {}
    for k, v in result._asdict().items():
        if v is not None:
            out[k] = v

    cleaned = json.dumps(out, cls=_SafeFallbackEncoder)
    return yaml.safe_dump(json.loads(cleaned), default_flow_style=False)
