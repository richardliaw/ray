import os
from datetime import datetime

from ray.tune.trial import Trial, TrialInfo
from ray.tune.result import DEFAULT_RESULTS_DIR, TRAINING_ITERATION
from ray.tune.logger import UnifiedLogger, Logger


class _ReporterHook(Logger):
    def __init__(self, tune_reporter):
        self.tune_reporter = tune_reporter

    def on_result(self, metrics):
        return self.tune_reporter(**metrics)


class TrackSession:
    """Manages results for a single session.

    Represents a single Trial in an experiment. This is automatically
    created when using ``tune.run``.

    Attributes:
        trial_name (str): Custom trial name.
        experiment_dir (str): Directory where results for all trials
            are stored. Each session is stored into a unique directory
            inside experiment_dir.
        upload_dir (str): Directory to sync results to.
        trial_config (dict): Parameters that will be logged to disk.
        _tune_reporter (StatusReporter): For rerouting when using Tune.
            Will not instantiate logging if not None.
    """

    def __init__(self,
                 trial_name=None,
                 experiment_dir=None,
                 upload_dir=None,
                 trial_config=None,
                 _tune_reporter=None):
        self._experiment_dir = None
        self._logdir = None
        self._upload_dir = None
        self.trial_config = None
        self._iteration = -1
        self._trial_info = None
        self.is_tune_session = bool(_tune_reporter)
        if self.is_tune_session:
            self._logger = _ReporterHook(_tune_reporter)
            self._logdir = _tune_reporter.logdir
            self._trial_info = _tune_reporter.trial_info
        else:
            trial_id = Trial.generate_id()
            self._trial_info = TrialInfo(
                trial_id=trial_id, trial_name=trial_name or trial_id)
            self._initialize_logging(experiment_dir, upload_dir, trial_config)

    def _initialize_logging(self,
                            experiment_dir=None,
                            upload_dir=None,
                            trial_config=None):
        if upload_dir:
            raise NotImplementedError("Upload Dir is not yet implemented.")

        # TODO(rliaw): In other parts of the code, this is `local_dir`.
        if experiment_dir is None:
            experiment_dir = os.path.join(DEFAULT_RESULTS_DIR, "default")

        self._experiment_dir = os.path.expanduser(experiment_dir)

        # TODO(rliaw): Refactor `logdir` to `trial_dir`.
        self._logdir = Trial.create_logdir(self.trial_name,
                                           self._experiment_dir)
        self._upload_dir = upload_dir
        self.trial_config = trial_config or {}

        # misc metadata to save as well
        self.trial_config["trial_id"] = self.trial_id
        self._logger = UnifiedLogger(self.trial_config, self._logdir)

    def log(self, **metrics):
        """Logs all named arguments specified in `metrics`.

        This will log trial metrics locally, and they will be synchronized
        with the driver periodically through ray.

        Arguments:
            metrics: named arguments with corresponding values to log.
        """
        self._iteration += 1
        # TODO: Implement a batching mechanism for multiple calls to `log`
        #     within the same iteration.
        metrics_dict = metrics.copy()
        metrics_dict.update({"trial_id": self.trial_info.trial_id})

        # TODO: Move Trainable autopopulation to a util function
        metrics_dict.setdefault(TRAINING_ITERATION, self._iteration)
        self._logger.on_result(metrics_dict)

    def close(self):
        """Closes loggers.

        No need to call this when using ``tune.run``.
        """
        self.trial_config["trial_completed"] = True
        self.trial_config["end_time"] = datetime.now().isoformat()
        # TODO(rliaw): Have Tune support updated configs
        self._logger.update_config(self.trial_config)
        self._logger.flush()
        self._logger.close()

    @property
    def logdir(self):
        """Trial logdir (subdir of given experiment directory)"""
        return self._logdir

    @property
    def trial_name(self):
        """Trial name for the corresponding trial of this Trainable.

        This is not set if not using Tune.
        """
        return self._trial_info.trial_name

    @property
    def trial_id(self):
        """Trial id for the corresponding trial of this Trainable.

        This is not set if not using Tune.
        """
        return self._trial_info.trial_id
