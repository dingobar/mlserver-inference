from abc import ABC, abstractmethod
from typing import Sequence
from mlserver_inference_pipeline.models import Prediction


class AbstractPredictionDestination(ABC):
    @abstractmethod
    def write(self, predictions: Sequence[Prediction]) -> None:
        pass
