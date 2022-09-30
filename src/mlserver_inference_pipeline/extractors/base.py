from abc import ABC, abstractmethod
from typing import Sequence

from pydantic import BaseSettings

from mlserver_inference_pipeline.models import FeatureSet


class AbstractFeatureExtractor(ABC):
    @abstractmethod
    def extract(self) -> FeatureSet:
        pass
