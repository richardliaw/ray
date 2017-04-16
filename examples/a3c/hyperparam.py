from multimodel import run_multimodel_experiment
import os
import json
#
# def run_multimodel_experiment(exp_count=1, num_workers=10, adam=False,
                    # sync=10, learning_rate=1e-4, infostr="", addr_info=None):

def save_results(exp_results):
    fdir = exp_results["exp_string"]
    try:
        os.makedirs(fdir)
    except Exception:
        pass
    with open(fdir + time.time() + ".json", "w") as f:
        json.dump(exp_results, f)
    print("Done")

def main():
    ray.init(redirect_output=True)
    all_experiments = []
    for repeat in range(3):
        for sync in [10, 30, 100]:
            for learning_rate in [10**(-x) for x in range(2, 5)]
                all_experiments.append(run_multimodel_experiment.remote())

    while len(all_experiments):
        done, all_experiments = ray.wait(all_experiments)
        results = ray.get(done)
        save_results(results)

if __name__ == '__main__':
    main()