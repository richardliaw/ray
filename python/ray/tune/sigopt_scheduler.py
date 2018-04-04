from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import numpy as np
try:
    import sigopt as sgo
except Exception as e:
    sgo = None

from ray.tune.trial import Trial
from ray.tune.error import TuneError
from ray.tune.trial_scheduler import TrialScheduler, FIFOScheduler
from ray.tune.config_parser import make_parser
from ray.tune.variant_generator import to_argv


class SigOptScheduler(FIFOScheduler):
    """FIFOScheduler that uses SigOpt to provide trial suggestions.

    Requires SigOpt to be installed. Variant generation will be limited as
    the hyperparameter configuration must be specified using SigOpt primitives.

    Parameters:
        max_concurrent (int | None): Number of maximum concurrent trials.
            If None, then trials will be queued only if resources
            are available.
        reward_attr (str): The TrainingResult objective value attribute.
            This refers to an increasing value, which is internally negated
            when interacting with HyperOpt. Suggestion procedures
            will use this attribute.

    Examples:
        >>> space = {'param': hp.uniform('param', 0, 20)}
        >>> config = {"my_exp": {
                          "run": "exp",
                          "repeat": 5,
                          "config": {"space": space}}}
        >>> run_experiments(config, scheduler=SigOptScheduler())
    """

    def __init__(self, key, max_concurrent=10,
                 reward_attr="episode_reward_mean"):
        assert sgo is not None, "SigOpt must be installed!"
        assert (type(max_concurrent) is int and max_concurrent > 0)
        self._max_concurrent = max_concurrent  # NOTE: this is modified later
        self._reward_attr = reward_attr
        self._sigopt_key = key
        self._experiment = None

    def add_experiment(self, experiment, trial_runner):
        """Tracks one experiment.

        Will error if one tries to track multiple experiments.
        """
        assert self._experiment is None, "SigOpt only tracks one experiment!"
        self._experiment = experiment

        self._output_path = experiment.name
        spec = copy.deepcopy(experiment.spec)

        # Set Scheduler field, as Tune Parser will default to FIFO
        assert spec.get("scheduler") in [None, "SigOpt"], "Incorrectly " \
            "specified scheduler!"
        spec["scheduler"] = "SigOpt"

        if "env" in spec:
            spec["config"] = spec.get("config", {})
            spec["config"]["env"] = spec["env"]
            del spec["env"]

        space = spec["config"]["space"]
        del spec["config"]["space"]

        self.parser = make_parser()
        self.args = self.parser.parse_args(to_argv(spec))
        self.args.scheduler = "SigOpt"
        self.default_config = copy.deepcopy(spec["config"])


        self.conn = Connection(client_token=self._sigopt_key)
        self._sgo_experiment = self.conn.experiments().create(
            name='Tune Experiment ({})'.format(name),
            parameters=parameters,
            parallel_bandwidth=self._max_concurrent,
        )
        self._num_trials_left = self.args.repeat

        if type(self._max_concurrent) is int:
            self._max_concurrent = min(self._max_concurrent, self.args.repeat)

        self.rstate = np.random.RandomState()
        self.trial_generator = self._trial_generator()
        self._add_new_trials_if_needed(trial_runner)

    def _trial_generator(self):
        while self._num_trials_left > 0:
            new_cfg = copy.deepcopy(self.default_config)
            suggestion = self.conn.experiments(
                self.experiment.id).suggestions().create()
            suggested_config = dict(suggestion.assignments)
            new_cfg.update(suggested_config)

            kv_str = "_".join(["{}={}".format(k, str(v)[:5])
                               for k, v in sorted(suggested_config.items())])
            experiment_tag = "{}_{}".format(suggestion.id, kv_str)

            # Keep this consistent with tune.variant_generator
            trial = Trial(
                trainable_name=self.args.run,
                config=new_cfg,
                local_dir=os.path.join(self.args.local_dir, self._output_path),
                experiment_tag=experiment_tag,
                resources=self.args.trial_resources,
                stopping_criterion=self.args.stop,
                checkpoint_freq=self.args.checkpoint_freq,
                restore_path=self.args.restore,
                upload_dir=self.args.upload_dir,
                max_failures=self.args.max_failures)

            self.suggestions[trial] = suggestion
            self._num_trials_left -= 1
            yield trial

    def on_trial_error(self, trial_runner, trial):
        self.on_trial_remove(trial_runner, trial)

    def on_trial_remove(self, trial_runner, trial):
        raise NotImplementedError

    def on_trial_complete(self, trial_runner, trial, result):
        suggestion = self.suggestions.pop(trial)

        self.conn.experiments(self.experiment.id).observations().create(
            suggestion=suggestion.id,
            value=getattr(result, self._reward_attr),
        )

    def choose_trial_to_run(self, trial_runner):
        self._add_new_trials_if_needed(trial_runner)
        return FIFOScheduler.choose_trial_to_run(self, trial_runner)

    def _add_new_trials_if_needed(self, trial_runner):
        """Checks if there is a next trial ready to be queued.

        This is determined by tracking the number of concurrent
        experiments and trials left to run. If self._max_concurrent is None,
        scheduler will add new trial if there is none that are pending.
        """
        pending = [t for t in trial_runner.get_trials()
                   if t.status == Trial.PENDING]
        if self._num_trials_left <= 0:
            return

        if len(self.suggestions) < self._max_concurrent:
            try:
                trial_runner.add_trial(next(self.trial_generator))
            except StopIteration:
                break
