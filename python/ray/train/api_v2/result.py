from typing import Dict, Any, List


class Result:
    def __init__(self, name: str, metrics: Dict[str, Any], checkpoint: Any):
        self.name = name
        self.metrics = metrics
        self.checkpoint = checkpoint

    def __repr__(self):
        return f"<Result name={self.name}, checkpoint={self.checkpoint}>"


class ResultGrid:
    def __init__(self, results: List[Result]):
        self.results = results
