from pydantic import BaseModel
from datetime import datetime


class DatasetMetadata(BaseModel):
    filename: str
    original_filename: str
    rows: int
    columns: int
    column_names: list[str]
    data_types: dict
    file_size: int
    uploaded_at: datetime