from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Sequence

import httpx

from mlserver_inference_pipeline.destinations.base import AbstractPredictionDestination
from mlserver_inference_pipeline.extractors.base import AbstractFeatureExtractor
from mlserver_inference_pipeline.kserve import KserveDataType
from mlserver_inference_pipeline.models import FeatureSet, KserveInferenceResponse, PredictionRecord


class AbstractPredictor(ABC):
    @abstractmethod
    def predict(self) -> None:
        pass


class MlserverPredictor:
    def __init__(
        self,
        mlserver_base_url: str,
        mlserver_model_name: str,
        feature_extractor: AbstractFeatureExtractor,
        prediction_destination: AbstractPredictionDestination,
    ) -> None:
        self.model_infer_url = (
            f"{mlserver_base_url.strip('/')}/v2/models/{mlserver_model_name}/infer"
        )
        self.feature_extractor = feature_extractor
        self.prediction_destination = prediction_destination

    def _call_model(self, features: FeatureSet) -> Sequence[PredictionRecord]:
        now = datetime.now(timezone.utc)
        r = httpx.post(
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
        r.raise_for_status()
        response_records = KserveInferenceResponse.parse_raw(r.read())

        prediction_records = []
        for index, output in enumerate(response_records.outputs[0].data):
            prediction_records.append(
                PredictionRecord(
                    input_ref=features.records[index].ref,
                    value=output,
                    model_version=response_records.model_version,
                    timestamp=now,
                )
            )
        return prediction_records

    def predict(self) -> None:
        features = self.feature_extractor.extract()
        predictions = self._call_model(features)
        self.prediction_destination.write(predictions)
