import numpy as np


def shard(array, num):
    rets = np.array_split(array, num)
    return rets