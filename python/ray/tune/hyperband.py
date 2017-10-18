# you need to write the following hooks for your custom problem
from problem import get_random_hyperparameter_configuration, run_then_return_val_loss

class Bracket():
    def __init__(self, initial_trials):
        self._total_resources = 0
        self._completed_resources = 0
        self.cur_trials = {trial: None for trial in initial_trials}
        n = int(ceil( (B / max_iter) * eta**s/ (s+1) )) # initial number of configurations
        r = max_iter * eta**(-s) # initial number of iterations to run configurations for

    def sucessive_halving(self, n, r, s):
        for i in range(s+1):
            # Run each of the n_i configs for r_i iterations and keep best n_i/eta
            val_losses = [run_then_return_val_loss(num_iters=r_i, hyperparameters=t) for t in T]

            n_i /= eta
            r_i *= eta
            T = [ T[i] for i in argsort(val_losses)[:int(n_i)] ]

    def update(self, trial, result):
        # if trial stopped or error, remove from bracket
        # else
        self.cur_trials[trial] = result

    def completion_percentage(self):
        return self._completed_resources / self._total_resources


class HyperBand(TrialRunner):

    """ Right now, this does NOT provide Tensorflow GPU support."""

    def __init__(self, max_iter, unit, eta=3, max_cycles=None):
        super(HyperBand, self).__init__(self)
        max_iter = 81  # maximum iterations/epochs per configuration
        min_iter = 1
        self.eta = eta # defines downsampling rate (default=3)
        logeta = lambda x: log(x)/log(eta)

        # number of unique executions of Successive Halving
        s_max_1 = int(logeta(max_iter)) + 1
        next_s = range(s_max_1)

        B = s_max_1 * max_iter  # total number of iterations (without reuse) per execution of Succesive Halving (n,r)
        #assign int(ceil( (B / max_iter) * eta**s/ (s+1) )) to S different brackets
        self.brackets = []
        #self.trial_bracket = {t: bracket for t in trial}


    def step(self):
        """Runs one step of the trial event loop.

        Callers should typically run this method repeatedly in a loop. They
        may inspect or modify the runner's state in between calls to step().

        Assume no general gpu support since nvidia doesn't release GPUs...
        """

        assert self._committed_resources.gpus == 0
        super(HyperBand, self).step()

    def _get_runnable(self):
        """Returns next trial from bracket. Sorting by percentage of completion
        as a fair scheduling mechanism.
        """

        for bracket in sorted(self.brackets, key=lambda b: b.completion_percentage()):
            for trial in bracket.cur_trials:
                if (trial.status == Trial.PENDING and
                        self._has_resources(trial.resources)):
                    return trial
        return None

    def _process_events(self):
        """Implements hyperband early stopping."""
        [result_id], _ = ray.wait(self._running.keys())
        trial = self._running[result_id]
        del self._running[result_id]
        try:
            result = ray.get(result_id)
            print("result", result)
            trial.last_result = result

            if trial.should_stop(result):
                self._stop_trial(trial)
            else:
                bracket = self.trial_bracket[trial]
                bracket.update(trial)
                if bracket.should_continue(trial):
                    # Note: This implements checkpoint in a blocking manner
                    if trial.should_checkpoint():
                        trial.checkpoint()
                    self._running[trial.train_remote()] = trial
                else:
                    trial.pause()
                if bracket.iteration_done():
                    bad_trials = bracket.successive_halving()
                    for t in bad_trials:
                        self._stop_trial(trial)
        except:
            print("Error processing event:", traceback.format_exc())
            if trial.status == Trial.RUNNING:
                self._stop_trial(trial, error=True)

    def _stop_trial(self, trial, error=False):
        del self.trial_bracket[trial]
        super(HyperBand, self)._stop_trial(trial, error)
