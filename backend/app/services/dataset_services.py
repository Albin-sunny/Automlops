import os
import shutil
import pandas as pd
from datetime import datetime
from fastapi import UploadFile, HTTPException
from app.repositories.dataset_repositories import save_dataset
from app.repositories.dataset_repositories import get_all_datasets
from app.repositories.dataset_repositories import get_dataset_by_id
from app.repositories.dataset_repositories import delete_dataset

UPLOAD_FOLDER = "datasets"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


async def process_dataset(file: UploadFile):

    # Validate file
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are allowed."
        )

    # Save file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Read CSV
    df = pd.read_csv(file_path)

    # Extract metadata
    rows = len(df)
    columns = len(df.columns)

    column_names = list(df.columns)

    data_types = {
        col: str(dtype)
        for col, dtype in df.dtypes.items()
    }

    file_size = os.path.getsize(file_path)

    metadata = {
        "message": "Dataset uploaded successfully",
        "filename": file.filename,
        "path": file_path,
        "rows": rows,
        "columns": columns,
        "column_names": column_names,
        "data_types": data_types,
        "file_size": file_size,
        "uploaded_at": datetime.utcnow()
    }

    dataset_id = await save_dataset(metadata)

    metadata["dataset_id"] = dataset_id

    return metadata


async def fetch_all_datasets():

    datasets = await get_all_datasets()

    return datasets


async def fetch_dataset_by_id(dataset_id: str):

    dataset = await get_dataset_by_id(dataset_id)

    return dataset


async def remove_dataset(dataset_id: str):

    dataset = await get_dataset_by_id(dataset_id)

    if not dataset:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found"
        )

    # Delete file
    file_path = dataset["path"]

    if os.path.exists(file_path):
        os.remove(file_path)

    # Delete MongoDB document
    deleted = await delete_dataset(dataset_id)

    return {
        "message": "Dataset deleted successfully",
        "deleted": deleted
    }
