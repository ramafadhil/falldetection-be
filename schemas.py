from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class AlertBase(BaseModel):
    device_id: int
    confidence: float

class AlertCreate(AlertBase):
    pass

class AlertResponse(BaseModel):
    id: int
    device_id: int
    confidence: float
    event_timestamp: datetime
    thumbnail_path: Optional[str] = None

    class Config:
        from_attributes = True