from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

import numpy as np 
from datetime import datetime
import cProfile, pstats, io
import os

def timestamp():
    return datetime.now().timestamp()

def time_string():
    return datetime.now().strftime("%Y%m%d_%H_%M_%f")

def try_makedirs(dir_name):
    # os.makedirs(dir_name)
    try:
        os.makedirs(dir_name)
    except Exception:
        print("Directory already made")

class Profiler(object):
    def __init__(self):
        self.pr = cProfile.Profile()
        pass

    def __enter__(self):
        self.pr.enable()

    def __exit__(self ,type, value, traceback):
        self.pr.disable()
        s = io.StringIO()
        sortby = 'cumtime'
        ps = pstats.Stats(self.pr, stream=s).sort_stats(sortby)
        ps.print_stats(.2)
        print(s.getvalue())

def load_weights(fname):
    with open(fname, "rb") as f:
        parameters = pickle.load(f)
    return parameters

def save_weights(params, fdir="./progress/", fname="policy.pkl"):
    fdir = os.path.join(fdir, time_string())
    try_makedirs(fdir)
    fname = os.path.join(fdir, fname)
    with open(fname, "wb") as f:
        pickle.dump(params, f)
