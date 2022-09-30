from datetime import datetime, timezone
from typing import Sequence
from uuid import UUID, uuid4

from pydantic import BaseModel

FeatureType = Sequence[float | int]


class FeatureRecord(BaseModel):
    ref: UUID | str | int
    features: FeatureType


class FeatureSet(BaseModel):
    records: Sequence[FeatureRecord]
    columns: Sequence[str]


class PredictionRecord(BaseModel):
    id: UUID = uuid4()
    input_ref: UUID | str | int
    value: Sequence[float | int] | float | int
    model_version: str
    timestamp: datetime = datetime.now(timezone.utc)


class KserveInferenceResponseOutput(BaseModel):
    name: str
    shape: list[int]
    datatype: str
    data: list[float]


class KserveInferenceResponse(BaseModel):
    model_name: str
    model_version: str
    id: str
    outputs: list[KserveInferenceResponseOutput]
