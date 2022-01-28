import abc

import ray.data


class Model(abc.ABC):
    def predict(self, preprocessed_data: ray.data.Dataset) -> ray.data.Dataset:
        raise NotImplementedError
