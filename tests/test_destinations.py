from pathlib import Path
from random import uniform
import mlserver_inference_pipeline.predict as predict
import pytest
from mlserver_inference_pipeline.models import FeatureRecord, Prediction
from mlserver_inference_pipeline.destinations.csv import CsvPredictionDestination
import tempfile


@pytest.fixture
def dummy_prediction_set():
    return [
        Prediction(
            input=FeatureRecord(ref=i, features=[1, 2, 3]),
            value=uniform(0, 1),
            model_version="test",
        )
        for i in range(100)
    ]


def test_destination_csv(
    monkeypatch: pytest.MonkeyPatch, dummy_prediction_set: list[Prediction]
):
    with tempfile.TemporaryDirectory() as f:
        output_file = Path(f) / "myfile.csv"
        monkeypatch.setenv("DESTINATION_CSV_OUTPATH", str(output_file))
        CsvPredictionDestination().write(dummy_prediction_set)
        assert output_file.exists()
        assert output_file.stat().st_size > 10
