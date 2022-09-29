from pathlib import Path
from typing import Sequence
from mlserver_inference_pipeline.destinations.base import AbstractPredictionDestination
from pydantic import BaseSettings
from mlserver_inference_pipeline.models import PredictionRecord
import csv


class CsvPredictionDestinationConfig(BaseSettings):
    outpath: Path = Path.cwd() / "out.csv"

    class Config:
        env_prefix = "DESTINATION_CSV_"


class CsvPredictionDestination(AbstractPredictionDestination):
    def __init__(self) -> None:
        self.config = CsvPredictionDestinationConfig()

    def write(self, predictions: Sequence[PredictionRecord]) -> None:
        with open(self.config.outpath, "w") as f:
            writer = csv.DictWriter(f, predictions[0].dict().keys())
            writer.writeheader()
            writer.writerows([p.dict() for p in predictions])
