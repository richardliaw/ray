import pandas as pd
import numpy as np
from typing import *

from ray import serve

DataBatchType = Union[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]

@serve.deployment
class ModelWrapper:
    def __init__(self,
            chpkt: Checkpoint, preprocessor: bool = True):
        self.model = chkpt.load_model()
        if preprocessor:
           self.prep = chkpt.load_preprocessor()
        else:
           self.prep = NoopPreprocessor

    def __call__(self, data: DataBatchType) -> DataBatchType:
        return self.model(self.prep.transform_batch(data))


def test_serve():
    checkpoint = load_checkpoint_from_path()
    serve.start()
    ModelWrapper.deploy(checkpoint)