import pytest

import mlserver_inference_pipeline.models as models


@pytest.mark.parametrize(
    "inputs",
    [
        {
            "input_ref": "a",
            "value": 1.234,
            "model_version": "v1",
        },
        {
            "id": "8156f0b4-3973-4ef1-a5aa-48d63fc6bc95",
            "input_ref": 6,
            "value": 1.234,
            "model_version": "v1",
        },
        {
            "input_ref": "a",
            "value": 1.234,
            "model_version": "v1",
            "timestamp": "2020-01-01T00:00:00Z",
        },
    ],
)
def test_create_prediction_object(inputs: list[dict]):
    models.PredictionRecord(**inputs)


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
