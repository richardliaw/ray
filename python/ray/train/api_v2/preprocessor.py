import abc
from typing import List

import numpy as np
import pandas as pd

import ray.data
from ray.data.aggregate import Mean, Std


class Preprocessor(abc.ABC):
    def fit(self, dataset: ray.data.Dataset) -> "Preprocessor":
        raise NotImplementedError

    def transform(self, dataset: ray.data.Dataset) -> ray.data.Dataset:
        raise NotImplementedError

    def fit_transform(self, dataset: ray.data.Dataset) -> ray.data.Dataset:
        # Todo: optimize
        self.fit(dataset)
        return self.transform(dataset)


class Scaler(Preprocessor):
    def __init__(self, columns: List[str]):
        self.columns = columns
        self.stats = None

    def fit(self, dataset: ray.data.Dataset) -> "Preprocessor":
        aggregates = [Agg(col) for Agg in [Mean, Std] for col in self.columns]
        self.stats = dataset.aggregate(*aggregates)
        return self

    def transform(self, dataset: ray.data.Dataset) -> ray.data.Dataset:
        columns = self.columns
        stats = self.stats

        def _scale(df: pd.DataFrame):
            def column_standard_scaler(s: pd.Series):
                s_mean = stats[f"mean({s.name})"]
                s_std = stats[f"std({s.name})"]
                return (s - s_mean) / s_std

            df.loc[:, columns] = df.loc[:, columns].transform(
                column_standard_scaler)
            return df

        return dataset.map_batches(_scale, batch_format="pandas")

    def __repr__(self):
        return f"<Scaler columns={self.columns} stats={self.stats}>"


class Repartitioner(Preprocessor):
    def __init__(self, num_partitions: int):
        self.num_partitions = num_partitions

    def fit(self, dataset: ray.data.Dataset) -> "Preprocessor":
        return self

    def transform(self, dataset: ray.data.Dataset) -> ray.data.Dataset:
        return dataset.repartition(num_blocks=self.num_partitions)

    def __repr__(self):
        return f"<Repartitioner num_partitions={self.num_partitions}>"


class Chain(Preprocessor):
    def __init__(self, *preprocessors):
        self.preprocessors = preprocessors

    def fit(self, dataset: ray.data.Dataset) -> "Preprocessor":
        for preprocessor in self.preprocessors:
            preprocessor.fit(dataset)
        return self

    def transform(self, dataset: ray.data.Dataset) -> ray.data.Dataset:
        for preprocessor in self.preprocessors:
            dataset = preprocessor.transform(dataset)
        return dataset

    def __repr__(self):
        return (f"<Chain preprocessors=["
                f"{', '.join(str(p) for p in self.preprocessors)}"
                f"]>")


def test_transform_scaler():
    """Scale B and C, but not A"""
    num_items = 1_000
    col_a = np.random.normal(loc=4., scale=0.4, size=num_items)
    col_b = np.random.normal(loc=8., scale=0.7, size=num_items)
    col_c = np.random.normal(loc=7., scale=0.3, size=num_items)
    in_df = pd.DataFrame.from_dict({"A": col_a, "B": col_b, "C": col_c})

    ds = ray.data.from_pandas(in_df)

    scaler = Scaler(["B", "C"])
    scaler.fit(ds)
    transformed = scaler.transform(ds)
    out_df = transformed.to_pandas()

    assert in_df["A"].equals(out_df["A"])
    assert not in_df["B"].equals(out_df["B"])
    assert not in_df["C"].equals(out_df["C"])

    print(in_df)
    print(out_df)
