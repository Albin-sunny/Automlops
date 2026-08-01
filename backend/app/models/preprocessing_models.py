from pydantic import BaseModel
from datetime import datetime


class PreprocessingResult(BaseModel):
    dataset_id: str
    processed_file_path: str
    original_rows: int
    processed_rows: int
    missing_values_handled: bool
    duplicates_removed: int
    created_at: datetime