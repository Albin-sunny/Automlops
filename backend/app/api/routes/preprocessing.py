from fastapi import APIRouter, HTTPException
from bson import ObjectId

from app.database import mongodb
from app.services.preprocessing_services import preprocess_dataset

router = APIRouter()


@router.post("/{dataset_id}")
async def run_preprocessing(dataset_id: str):

    dataset = await mongodb.database.datasets.find_one(
        {"_id": ObjectId(dataset_id)}
    )

    if not dataset:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found"
        )

    result = await preprocess_dataset(
        dataset["path"],
        dataset_id,
        dataset["target_column"]
    )

    return result