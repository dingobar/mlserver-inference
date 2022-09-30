from mlserver_inference_pipeline.destinations.console import (
    ConsolePredictionDestination,
)
import pytest
from respx.router import MockRouter
from random import uniform
from httpx import Response
from mlserver_inference_pipeline.extractors.base import AbstractFeatureExtractor
from mlserver_inference_pipeline.destinations.base import AbstractPredictionDestination
from mlserver_inference_pipeline.predict import MlserverPredictor
from mlserver_inference_pipeline.models import FeatureRecord, FeatureSet


class MockExtractor(AbstractFeatureExtractor):
    def extract(self) -> FeatureSet:
        return FeatureSet(
            columns=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            records=[
                FeatureRecord(ref=i, features=[uniform(0, 1) for _ in range(10)])
                for i in range(10)
            ],
        )


@pytest.fixture
def mock_extractor():
    return MockExtractor()


@pytest.fixture
def console_destination():
    return ConsolePredictionDestination()


def test_mlserver_predictor(
    respx_mock: MockRouter,
    mock_extractor: AbstractFeatureExtractor,
    console_destination: AbstractPredictionDestination,
):
    dummy_model_url = "https://dummy.model.com/dummymodel/"
    mock = respx_mock.post(dummy_model_url + "v2/models/mlflow-model/infer")
    mock.return_value = Response(
        200,
        content=b'{"model_name":"mlflow-model","model_version":"runs:/60b9e37769544d27a96f8fc2797fed6f/model","id":"fe40825a-7a5b-4d00-acee-c97bfed27bfd","parameters":{"content_type":null,"headers":null},"outputs":[{"name":"output-1","shape":[10],"datatype":"FP64","parameters":null,"data":[233.01528477525025,220.4473218122873,220.4473218122873,220.4473218122873,220.4473218122873,222.3119047254526,222.46509959006508,220.4473218122873,220.55527737785096,220.4473218122873]}]}',
    )
    predictor = MlserverPredictor(
        dummy_model_url,
        "mlflow-model",
        mock_extractor,
        console_destination,
    )
    predictor.predict()
