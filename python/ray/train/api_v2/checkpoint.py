import abc
import os
import pickle
import tarfile
import tempfile
import time
from typing import Any, Optional

import ray
from ray.train.api_v2.preprocessor import Preprocessor
from ray.train.api_v2.model import Model


def _pack(dir: str):
    _, tmpfile = tempfile.mkstemp()
    with tarfile.open(tmpfile, "w:gz") as tar:
        tar.add(dir, arcname="")

    with open(tmpfile, "rb") as f:
        stream = f.read()

    return stream


def _unpack(stream: bytes, dir: str):
    _, tmpfile = tempfile.mkstemp()

    with open(tmpfile, "wb") as f:
        f.write(stream)

    with tarfile.open(tmpfile) as tar:
        tar.extractall(dir)


class ArtifactData:
    def __init__(self, data: Any):
        self.data = data


class ArtifactDirectory(ArtifactData):
    pass


class ArtifactFile(ArtifactData):
    pass


class ArtifactObject(ArtifactData):
    pass


class Artifact(abc.ABC):
    """Artifact interface"""
    pass


class ObjectStoreArtifact(Artifact):
    def __init__(self, obj_ref: ray.ObjectRef):
        self.obj_ref = obj_ref

    def _to_local_storage(self, path: str) -> "LocalStorageArtifact":
        return LocalStorageArtifact(path=path)

    def to_local_storage(self, path: str) -> "LocalStorageArtifact":
        data = ray.get(self.obj_ref)
        with open(path, "wb") as fp:
            pickle.dump(data, fp)
        return self._to_local_storage(path)


class LocalStorageArtifact(Artifact):
    def __init__(self, path: str):
        self.path = path

    def _to_object_store(self,
                         obj_ref: ray.ObjectRef) -> "ObjectStoreArtifact":
        return ObjectStoreArtifact(obj_ref=obj_ref)

    def to_object_store(self) -> "ObjectStoreArtifact":
        if os.path.isdir(self.path):
            data = ArtifactDirectory(_pack(self.path))
        else:
            with open(self.path, "r") as fp:
                data = ArtifactFile(fp.read())
        return self._to_object_store(ray.put(data))


class Checkpoint(abc.ABC):
    def load_model(self, **options) -> Model:
        """Load the model from storage."""
        raise NotImplementedError

    def load_preprocessor(self, **options) -> Preprocessor:
        """Returns fitted preprocessor."""
        raise NotImplementedError

    def as_callable_class(self, **options):
        raise NotImplementedError


class ObjectStoreCheckpoint(Checkpoint):
    def __init__(self, obj_ref: ray.ObjectRef):
        self.obj_ref = obj_ref

    def _to_local_storage(self, path: str) -> "LocalStorageCheckpoint":
        return LocalStorageCheckpoint(path=path)

    def to_local_storage(
            self, path: Optional[str] = None) -> "LocalStorageCheckpoint":
        if path is None:
            path = tempfile.mktemp()
        data = ray.get(self.obj_ref)
        if isinstance(data, ArtifactDirectory):
            _unpack(data.data, path)
        elif isinstance(data, ArtifactFile):
            with open(path, "wb") as fp:
                pickle.dump(data.data, fp)
        else:
            with open(path, "wb") as fp:
                pickle.dump(data, fp)
        return self._to_local_storage(path)

    def __repr__(self):
        return f"<ObjectStoreCheckpoint obj_ref={self.obj_ref}>"

    def __getstate__(self):
        state = self.__dict__.copy()
        obj_ref = state.pop("obj_ref", None)
        if obj_ref:
            data = ray.get(obj_ref)
        else:
            data = None

        state["_data"] = data
        return state

    def __setstate__(self, state):
        data = state.pop("_data", None)
        self.__dict__.update(state)

        if data:
            self.obj_ref = ray.put(data)


class LocalStorageCheckpoint(Checkpoint):
    def __init__(self, path: str):
        self.path = path

    def _to_object_store(self,
                         obj_ref: ray.ObjectRef) -> "ObjectStoreCheckpoint":
        return ObjectStoreCheckpoint(obj_ref=obj_ref)

    def to_object_store(self) -> "ObjectStoreCheckpoint":
        if os.path.isdir(self.path):
            data = ArtifactDirectory(_pack(self.path))
        else:
            with open(self.path, "r") as fp:
                data = ArtifactFile(fp.read())
        return self._to_object_store(obj_ref=ray.put(data))

    def __repr__(self):
        return f"<LocalStorageCheckpoint path={self.path}>"


class TrainCheckpoint(Checkpoint, Artifact, abc.ABC):
    def __init__(self,
                 metric: float,
                 preprocessor: Optional[Preprocessor] = None):
        self.metric = metric
        self.creation_time = time.time()
        self.creation_node = ray.util.get_node_ip_address()

        self.preprocessor = preprocessor

    def load_preprocessor(self, **options) -> Preprocessor:
        return self.preprocessor


class TrainLocalStorageCheckpoint(TrainCheckpoint, LocalStorageCheckpoint):
    def __init__(self,
                 path: str,
                 metric: float,
                 preprocessor: Optional[Preprocessor] = None):
        TrainCheckpoint.__init__(self, metric, preprocessor)
        LocalStorageCheckpoint.__init__(self, path)

    def _to_object_store(self, obj_ref: ray.ObjectRef):
        return TrainObjectStoreCheckpoint(
            obj_ref=obj_ref,
            metric=self.metric,
            preprocessor=self.preprocessor)


class TrainObjectStoreCheckpoint(TrainCheckpoint, ObjectStoreCheckpoint):
    def __init__(self,
                 obj_ref: ray.ObjectRef,
                 metric: float,
                 preprocessor: Optional[Preprocessor] = None):
        TrainCheckpoint.__init__(self, metric, preprocessor)
        ObjectStoreCheckpoint.__init__(self, obj_ref)

    def _to_local_storage(self, path: str) -> "TrainLocalStorageCheckpoint":
        return TrainLocalStorageCheckpoint(
            path=path, metric=self.metric, preprocessor=self.preprocessor)
