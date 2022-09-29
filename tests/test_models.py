import mlserver_inference_pipeline.models as models
import pytest


@pytest.mark.parametrize(
    "inputs",
    [
        {
            "input": {"ref": "a", "features": [1, 2, 3]},
            "value": 1.234,
            "model_version": "v1",
        },
        {
            "id": "8156f0b4-3973-4ef1-a5aa-48d63fc6bc95",
            "input": {"ref": "a", "features": [1, 2, 3]},
            "value": 1.234,
            "model_version": "v1",
        },
        {
            "input": {"ref": "a", "features": [1.1, 2.2, 3.3]},
            "value": 1.234,
            "model_version": "v1",
            "timestamp": "2020-01-01T00:00:00Z",
        },
    ],
)
def test_create_prediction_object(inputs: list[dict]):
    models.Prediction(**inputs)


@pytest.mark.parametrize(
    "inputs",
    [
        {
            "records": [{"ref": "foo", "features": [0, 0, 0]}],
            "columns": ["a", "b", "c"],
        },
    ],
)
def test_create_featureset_object(inputs: list[dict]):
    models.FeatureSet(**inputs)
