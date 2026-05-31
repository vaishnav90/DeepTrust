from pydantic import BaseModel
from datetime import date, time
from typing import Optional

class DetectionLog(BaseModel):
    id: Optional[int] = None
    date: date
    hour: time
    classification: Optional[str] = None  # "Deepfake" | "Bonafide"
    score: Optional[float] = None  # deepfake risk 0..100 (higher = more fake)
    fidelity: Optional[float] = None  # confidence 0..100 (higher = more confident)
    
    class Config:
        json_encoders = {
            date: lambda v: v.isoformat(),
            time: lambda v: v.isoformat()
        }
