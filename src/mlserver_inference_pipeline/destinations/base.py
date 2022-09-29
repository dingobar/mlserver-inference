from abc import ABC, abstractmethod
from typing import Sequence
from mlserver_inference_pipeline.models import PredictionRecord


class AbstractPredictionDestination(ABC):
    @abstractmethod
    def write(self, predictions: Sequence[PredictionRecord]) -> None:
        pass
