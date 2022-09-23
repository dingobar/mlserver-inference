from datetime import datetime, timezone
from random import choice, uniform
import random
import string
from typing import Hashable
from mlserver_inference_pipeline.extractors.base import AbstractFeatureExtractor
from mlserver_inference_pipeline.models import FeatureSet
from pydantic import BaseSettings


class RandomFeatureExtractorConfig(BaseSettings):

    seed: Hashable = datetime.now(timezone.utc)
    n_features: int
    length: int = 1

    class Config:
        env_prefix = "EXTRACTOR_RANDOM_"


class RandomFeatureExtractor(AbstractFeatureExtractor):
    def __init__(self) -> None:
        self.config = RandomFeatureExtractorConfig()
        random.seed(self.config.seed)

    def extract(self) -> FeatureSet:
        def randomword(length):
            letters = string.ascii_lowercase
            return "".join(choice(letters) for i in range(length))

        features = [
            [uniform(0, 1) for _ in range(self.config.n_features)]
            for _ in range(self.config.length)
        ]
        columns = [randomword(10) for _ in range(self.config.n_features)]
        return FeatureSet(features=features, columns=columns)
