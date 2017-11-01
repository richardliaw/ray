import ray
ray.init()
from ray.rllib.a3c import A3CAgent
a = A3CAgent("PongDeterministic-v4", {"num_workers": 4,})
for i in range(50):
    a.train()

