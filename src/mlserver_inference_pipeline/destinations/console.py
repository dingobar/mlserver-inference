from typing import Sequence

from mlserver_inference_pipeline.destinations.base import AbstractPredictionDestination
from mlserver_inference_pipeline.models import PredictionRecord


class ConsolePredictionDestination(AbstractPredictionDestination):
    def write(self, predictions: Sequence[PredictionRecord]) -> None:
        for prediction in predictions:
            print(prediction)
