from datetime import datetime, timezone
from uuid import UUID, uuid4
from pydantic import BaseModel, validator
from typing import Sequence

FeatureType = Sequence[float | int]


class FeatureSet(BaseModel):
    features: FeatureType | Sequence[FeatureType]
    columns: Sequence[str]


class Prediction(BaseModel):
    id: UUID = uuid4()
    ref: UUID | str | int
    value: Sequence[float | int] | float | int
    model_version: str
    timestamp: datetime = datetime.now(timezone.utc)
