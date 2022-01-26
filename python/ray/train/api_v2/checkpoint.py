import abc
import os
import pickle
import tarfile
import tempfile
import time
from typing import Union, Any, Optional, Tuple

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

    def to_local_storage(self, path: str) -> "LocalStorageArtifact":
        data = ray.get(self.obj_ref)
        with open(path, "wb") as fp:
            pickle.dump(data, fp)
        return LocalStorageArtifact(path=path)


class LocalStorageArtifact(Artifact):
    def __init__(self, path: str):
        self.path = path

    def to_object_store(self):
        if os.path.isdir(self.path):
            data = ArtifactDirectory(_pack(self.path))
        else:
            with open(self.path, "r") as fp:
                data = ArtifactFile(fp.read())


class MemoryArtifact(Artifact):
    def __init__(self, data: Union[ray.ObjectRef, Any]):
        self._future = None
        self.data = None

        if isinstance(data, ray.ObjectRef):
            self._future = data
        else:
            self.data = data

    @property
    def ready(self):
        return self._future is None

    def wait(self):
        if self.ready:
            return
        self.data = ray.get(self._future)

    def save(self, path: Optional[str] = None):
        pass

    def delete(self) -> None:
        """Delete artifact data."""
        self._future = None
        self.data = None

    def to_local_storage(self, path: str) -> "LocalStorageArtifact":
        self.save(path=path)
        return LocalStorageArtifact(path=path)


class LocalStorageArtifact(Artifact):
    def __init__(self, path: Union[str, Tuple[str], ray.ObjectRef]):
        self._future = None
        self.local_path = None
        self.cloud_path = None
        self.creation_node_ip = ray.util.get_node_ip_address()

        if isinstance(path, ray.ObjectRef):
            self._future = path
        else:
            self._parse_path(path)

    def _parse_path(self, path: Union[str, Tuple[str]]):
        if isinstance(path, str):
            path = (path, )

        for p in path:
            if is_cloud_target(p):
                self.cloud_path = p
            else:
                self.local_path = p

    @property
    def ready(self):
        return self._future is None

    def wait(self):
        if self.ready:
            return
        path = ray.get(self._future)
        self._parse_path(path=path)

    def download(self,
                 cloud_path: Optional[str] = None,
                 local_path: Optional[str] = None,
                 overwrite: bool = False) -> str:
        """Download artifact from cloud to local storage."""
        pass

    def upload(self,
               cloud_path: Optional[str] = None,
               local_path: Optional[str] = None,
               clean_before: bool = False):
        """Upload artifact from local storage to cloud."""
        pass

    def save(self, path: Optional[str] = None, force_download: bool = False):
        pass

    def delete(self) -> None:
        """Delete artifact from local path, cloud path, and creation node."""
        pass

    def to_memory(self) -> "MemoryArtifact":
        pass


class ObjectStoreArtifact(Artifact):
    pass


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
    pass


class TuneCheckpoint(Checkpoint, Artifact, abc.ABC):
    def __init__(self, metric: float):
        self.metric = metric
        self.creation_time = time.time()


class MemoryCheckpoint(MemoryArtifact, TuneCheckpoint):
    def __init__(self, data: Union[ray.ObjectRef, Any], metric: float):
        MemoryArtifact.__init__(self, data=data)
        TuneCheckpoint.__init__(self, metric=metric)

    def as_persisted(self) -> "PersistedCheckpoint":
        return self


class PersistedCheckpoint(PersistedArtifact, TuneCheckpoint):
    def __init__(self, path: Union[str, Tuple[str], ray.ObjectRef],
                 metric: float):
        PersistedArtifact.__init__(self, path=path)
        TuneCheckpoint.__init__(self, metric=metric)

    def as_persisted(self) -> "PersistedCheckpoint":
        return self
