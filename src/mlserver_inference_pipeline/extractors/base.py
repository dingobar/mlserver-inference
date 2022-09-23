from abc import ABC, abstractmethod
from typing import Sequence
from mlserver_inference_pipeline.models import FeatureSet
from pydantic import BaseSettings


class AbstractFeatureExtractor(ABC):
    @abstractmethod
    def extract(self) -> FeatureSet:
        pass
