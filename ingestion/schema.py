from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class IncidentDocument(BaseModel):
    doc_id: str
    doc_type: str  # incident | runbook | architecture
    service: str
    severity: str  # low | medium | high
    timestamp: datetime

    summary: str
    root_cause: Optional[str] = None
    resolution: Optional[str] = None
    lessons_learned: Optional[str] = None

    source_url: str
    is_synthetic: bool
