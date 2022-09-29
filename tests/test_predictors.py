from mlserver_inference_pipeline.destinations.console import (
    ConsolePredictionDestination,
)
import pytest
from random import uniform
from mlserver_inference_pipeline.extractors.base import AbstractFeatureExtractor
from mlserver_inference_pipeline.destinations.base import AbstractPredictionDestination
from mlserver_inference_pipeline.predict import MlserverPredictor
from mlserver_inference_pipeline.models import FeatureRecord, FeatureSet


class MockExtractor(AbstractFeatureExtractor):
    def extract(self) -> FeatureSet:
        return FeatureSet(
            columns=["a", "b", "c"],
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
    mock_extractor: AbstractFeatureExtractor,
    console_destination: AbstractPredictionDestination,
):
    # TODO: mock httpx to make this pass https://lundberg.github.io/respx/
    predictor = MlserverPredictor(
        "http://my-host.com", "mymodel", mock_extractor, console_destination
    )
    predictor.predict()
