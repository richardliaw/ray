import argparse
import numpy as np
import json
from collections import defaultdict
import time
from datetime import timedelta
import ray
import tensorflow as tf
from driver import Runner
from six.moves import queue
from runner import RunnerThread, process_rollout, UpdateThread
import sys, os
from misc import *
from envs import create_env
from LSTM import LSTMPolicy
from FC import FCPolicy
from csv import DictWriter
import threading
BATCH = 20

@ray.actor
class Training():
    def __init__(self, mid, num_workers=2, opt_type="adam", learning_rate=1e-4, env_name="PongDeterministic-v3", log_dir="/tmp/results/"):
        assert type(opt_type) == str, type(opt_type)
        try:
            os.makedirs(log_dir)
        except Exception as e:
            pass
        self.mid = mid
        self.env_name = env_name
        self.log_dir = log_dir
        self.log = None
        self.node = ray.services.get_node_ip_address()

        #inline defn needed
        @ray.actor(local=True)
        class Runner(object):
            """Actor object to start running simulation on workers.
                Gradient computation is also executed from this object."""
            def __init__(self, env_name, actor_id, logdir="./results/tf/", start=True):
                env = create_env(env_name)
                self.id = actor_id
                num_actions = env.action_space.n
                self.policy = LSTMPolicy(env.observation_space.shape, num_actions, actor_id)
                self.runner = RunnerThread(env, self.policy, BATCH)
                self.env = env
                self.logdir = logdir
                if start:
                    self.start()

            def pull_batch_from_queue(self):
                """ self explanatory:  take a rollout from the queue of the thread runner. """
                rollout = self.runner.queue.get(timeout=600.0)
                while not rollout.terminal:
                    try:
                        rollout.extend(self.runner.queue.get_nowait())
                    except queue.Empty:
                        break
                return rollout

            def start(self):
                summary_writer = tf.summary.FileWriter(os.path.join(self.logdir, "agent_%d" % self.id))
                self.summary_writer = summary_writer
                self.runner.start_runner(self.policy.sess, summary_writer)

            def compute_gradient(self, params, tasksub):
                _start = timestamp()
                self.policy.set_weights(params)
                rollout = self.pull_batch_from_queue()
                batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)
                gradient = self.policy.get_gradients(batch)
                end = timestamp()
                info = {"id": self.id,
                        "size": len(batch.a),
                        "node": ray.services.get_node_ip_address(),
                        "start_task": _start - tasksub,
                        "end": end,
                        "time": end - _start,
                        "results": rollout.final }
                return gradient, info
        ## end inline ddef
        self.agents = [Runner(env_name, i, log_dir) for i in range(int(num_workers))]
        self.num_w = num_workers

        env = create_env(self.env_name)
        self.policy = LSTMPolicy(env.observation_space.shape, env.action_space.n, 0, opt_hparams={"learning_rate": learning_rate, "type": opt_type})
        if opt_type == "adam":
            assert self.policy.opt.get_name() == "Adam"
  

    def get_log_dir(self):
        return self.log_dir
    
    def get_node(self):
        return self.node

    def write_results(self, timing):
        if self.log is None:
            print("writing row")
        self.log.writerow(timing)

    def train(self, steps_max):
        parameters = self.policy.get_weights()
        gradient_list = [agent.compute_gradient(parameters, timestamp()) for agent in self.agents]
        steps = 0
        obs = 0
        timing = defaultdict(list)
        training_info = {}
        results = []
        FULLSTART = timestamp()

        while steps < steps_max:
            _start = timestamp()
            done_id, gradient_list = ray.wait(gradient_list)
            gradient, info = ray.get(done_id)[0]
            if info['results']:
                results.extend(info['results'])
            _getwait = timestamp()
            self.policy.model_update(gradient)
            _update = timestamp()
            parameters = self.policy.get_weights()
            _endget = timestamp()
            steps += 1
            obs += info["size"]
            gradient_list.extend([self.agents[info["id"]].compute_gradient(parameters, timestamp())])
            _endsubmit = timestamp()
            timing["Task"].append(info["time"])
            timing["Task_start"].append(info["start_task"])
            timing["Task_end"].append(_getwait - info["end"])
            timing["1.Wait"].append(_getwait - _start)
            timing["2.Update"].append(_update - _getwait)
            timing["3.Weights"].append(_endget - _update)
            timing["4.Submit"].append(_endsubmit - _endget)
            timing["5.Total"].append(_endsubmit - _start)

        timing = {k: np.mean(v) for k, v in timing.items()}
        timing_str =  str(self.node) + " ".join(["%s: %f" % (k, v) for k, v in sorted(timing.items())])
        timing["results"] = results
        timing["timing_str"] = timing_str
        timing["node"] = self.node
        timing["mid"] = self.mid
        timing["duration"] = (timestamp(), FULLSTART)
        return self.get_weights_with_optimizer(), timing

    def set_weights_with_optimizer(self, weights):
        self.policy.set_weights_with_optimizer(weights)

    def get_weights_with_optimizer(self):
        return self.policy.get_weights_with_optimizer()

    def set_weights(self, weights):
        self.policy.set_weights(weights)

    def get_weights(self):
        return self.policy.get_weights()

def model_averaging(params, stats=None):
    loader = defaultdict(list)
    for param_dict in params:
        for k, v in param_dict.items():
            loader[k].append(v)
    return {k: np.mean(v, axis=0) for k ,v in loader.items()}

def best_model(params, stats):
    mean = [m for m, s in stats]
    print(mean)
    best = np.argmax(mean)
    print("Choosing %d..." % best)
    return params[best]

def drop_half(params, stats):
    mean = [m for m, s in stats]
    half = int(-len(mean) / 2)
    idx = np.argsort(mean)[half:]
    idx = idx.repeat(2)
    return [params[i] for i in idx]

def generate_path(params):
    return  "./results/" + "-".join(["%s%s" % (k, str(v)) for k, v in sorted(params.items())]) + "/"

def make_log(fdir, timing):
    fname = "%s.csv" % time_string()
    try_makedirs(fdir)
    log = DictWriter(open(os.path.join(fdir, fname), "w"), timing.keys())
    log.writeheader()
    return log

def run_multimodel_experiment(exp_count=1, num_workers=10, opt_type="adam",
                    sync=10, learning_rate=1e-4, infostr="", 
                    addr_info=None, aggr_param="average",
                    load=""):
    aggregation = aggregation_function(aggr_param)
    SYNC = sync
    experiments = [Training(i, num_workers, opt_type) for i in range(exp_count)]
    all_info = defaultdict(list)
    all_info["exp_count"] = exp_count
    all_info["sync"] = SYNC
    all_info["workers"] = num_workers
    all_info["batch"] = BATCH
    if load:
        # WEIGHTS ARE NOT LOADED WITH OPTIMIZER
        new_params = load_weights(fname)
        ray.get([e.set_weights(new_params) for i, e in enumerate(experiments)])

    _start = time.time()
    new_params = ray.get(experiments[0].get_weights_with_optimizer())
    counter = 0
    itr = 0
    log = None
    logdir_params = {"wrkr_": exp_count, "aggr": aggr_param, "sync_": SYNC}
    while time.time() - _start < 1200:
        if type(new_params) == list:
            # If we want to use different type of models
            assert len(new_params) == exp_count
            ray.get([e.set_weights_with_optimizer(param) for e, param in zip(experiments, new_params)])
        else:
            ray.get([e.set_weights_with_optimizer(new_params) for i, e in enumerate(experiments)])
        __t = time.time()
        return_vals = ray.get([e.train(SYNC) for i, e in enumerate(experiments)])
        
        print("%d steps: %f time..." % (SYNC, time.time() - __t))
        params, information = zip(*return_vals)
        stats = [(np.mean(x["results"]), np.std(x["results"])) for x in information]
        for tup in information:
            print(tup["timing_str"])
            del tup["timing_str"]
            if log is None:
                fdir = generate_path(logdir_params)
                log = make_log(fdir,  tup)
            log.writerow(tup)
        all_info["stats"].append(stats)
        all_info["TS"].append((time.time() - _start))
        time_str = str(timedelta(seconds=time.time() - _start))
        print("Time elapsed: " + time_str)
        print("Model performance: \n" + "\n".join(["%d -- Mean: %.4f | Std: %.4f" % (i, m, s) for i, (m, s) in enumerate(stats)])) 
        new_params = aggregation(params, stats)
        itr += 1
        # new_params = best_model(params, stats)
    fdir = generate_path(logdir_params)
    try_makedirs(fdir)
    with open(fdir + time_string() + ".json", "w") as f:
        json.dump(all_info, f)
    return all_info

def aggregation_function(val):
    if val == "average":
        return model_averaging
    if val == "drop_half":
        return drop_half
    if val == "best":
        return best_model 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the multi-model learning example.")
    parser.add_argument("--num-experiments", default=1, type=int, help="The number of training experiments")
    parser.add_argument("--runners", default=12, type=int, help="Number of simulations")
    parser.add_argument("--lr", default=1e-4, type=float, help="LearningRate")
    parser.add_argument("--type", default="adam", type=str, help="Type of Optimizer")
    parser.add_argument("--sync", default=10, type=int, help="Sync Step")
    parser.add_argument("--addr", default=None, type=str, help="The Redis address of the cluster.")
    parser.add_argument("--aggr", default="average", type=str, help="Aggregation Technique")
    parser.add_argument("--info", default="", type=str, help="Information for file name")
    parser.add_argument("--load", default="", type=str, help="Load pretrained weights")
    opts = parser.parse_args(sys.argv[1:])
    if opts.addr:
        address_info = ray.init(redirect_output=True, redis_address=opts.addr)
    else:
        address_info = ray.init(redirect_output=False, num_workers=1 )
    # addr_info_id = ray.put(address_info)
    exp_results = run_multimodel_experiment(opts.num_experiments, 
                        num_workers=opts.runners, 
                        sync=opts.sync,
                        learning_rate=opts.lr,
                        opt_type=opts.type,
                        infostr=opts.info,
                        aggr_param=opts.aggr,
                        addr_info=address_info,
                        load=opts.load)
    # save_results(exp_results)
