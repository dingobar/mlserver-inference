from datetime import datetime, timezone
from uuid import UUID, uuid4
from pydantic import BaseModel, validator
from typing import Sequence

FeatureType = Sequence[float | int]


class FeatureRecord(BaseModel):
    ref: UUID | str | int
    features: FeatureType


class FeatureSet(BaseModel):
    records: Sequence[FeatureRecord]
    columns: Sequence[str]


class Prediction(BaseModel):
    id: UUID = uuid4()
    input: FeatureRecord
    value: Sequence[float | int] | float | int
    model_version: str
    timestamp: datetime = datetime.now(timezone.utc)
