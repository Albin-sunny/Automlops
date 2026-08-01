from pydantic import BaseModel
from datetime import datetime


class ProfilingResult(BaseModel):
    dataset_id: str
    rows: int
    columns: int
    missing_values: dict
    duplicate_rows: int
    data_types: dict
    numeric_summary: dict
    quality_score: float
    created_at: datetime