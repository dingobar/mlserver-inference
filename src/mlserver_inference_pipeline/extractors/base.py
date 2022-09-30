from abc import ABC, abstractmethod

from mlserver_inference_pipeline.models import FeatureSet


class AbstractFeatureExtractor(ABC):
    @abstractmethod
    def extract(self) -> FeatureSet:
        pass
