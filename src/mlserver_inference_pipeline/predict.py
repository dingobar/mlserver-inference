from abc import ABC, abstractmethod
from ast import Str
from random import randint
from typing import Sequence
from mlserver_inference_pipeline.kserve import KserveDataType
from mlserver_inference_pipeline.models import FeatureSet, PredictionRecord
import httpx
from mlserver_inference_pipeline.extractors.base import AbstractFeatureExtractor
from mlserver_inference_pipeline.destinations.base import AbstractPredictionDestination
from mlserver_inference_pipeline.models import FeatureType


class AbstractPredictor(ABC):
    @abstractmethod
    def predict(self) -> Sequence[PredictionRecord]:
        pass


class MlserverPredictor:
    def __init__(
        self,
        mlserver_base_url: str,
        mlserver_model_name: Str,
        feature_extractor: AbstractFeatureExtractor,
        prediction_destination: AbstractPredictionDestination,
    ) -> None:
        self.model_infer_url = (
            f"{mlserver_base_url.strip('/')}/v2/models/{mlserver_model_name}/infer"
        )
        self.feature_extractor = feature_extractor
        self.prediction_destination = prediction_destination

    def _call_model(self, features: FeatureSet) -> Sequence[PredictionRecord]:
        httpx.post(
            self.model_infer_url,
            json={
                "inputs": [
                    {
                        "name": "input",
                        "shape": [-1, len(features.columns)],
                        "datatype": KserveDataType.FP64,
                        "data": [a.features for a in features.records],
                    }
                ]
            },
        )

    def predict(self) -> Sequence[PredictionRecord]:
        features = self.feature_extractor.extract()
        predictions = self._call_model(features)
        self.prediction_destination.write(predictions)
