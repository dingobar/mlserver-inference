import pytest
from mlserver_inference_pipeline.extractors.random import RandomFeatureExtractor


def test_extractor_random(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EXTRACTOR_RANDOM_N_FEATURES", "360")
    monkeypatch.setenv("EXTRACTOR_RANDOM_SEED", "reproduce")
    extractor = RandomFeatureExtractor()
    result = extractor.extract()

    assert len(result.features[0]) == 360
    assert len(result.columns) == 360
    # Setting the seed should give reproducible results
    assert result.features[0][0] == 0.7501699608094758
    assert result.columns[0] == "henfafsivi"
