from abc import ABC, abstractmethod
from ast import Str
from random import randint
from typing import Sequence
from mlserver_inference_pipeline.models import Prediction
import httpx
from mlserver_inference_pipeline.extractors.base import AbstractFeatureExtractor
from mlserver_inference_pipeline.destinations.base import AbstractPredictionDestination
from mlserver_inference_pipeline.models import FeatureType


class MlserverPredictor:
    def __init__(
        self,
        mlserver_base_url: str,
        mlserver_model_name: Str,
        feature_extractor: AbstractFeatureExtractor,
        prediction_destination: AbstractPredictionDestination,
    ) -> None:
        self.model_infer_url = (
            f"{mlserver_base_url.strip('/')}/v2/models/{mlserver_model_name}"
        )
        self._get_endpoint_metadata()

    def _get_endpoint_metadata(self):
        r = httpx.get(self.model_infer_url)
        return r.json()

    def predict(self, features: Sequence[FeatureType]) -> Sequence[Prediction]:
        pass
