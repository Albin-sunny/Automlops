from fastapi import APIRouter, HTTPException

from app.database import mongodb
from app.services.training_services import train_models

from bson import ObjectId


router = APIRouter()


@router.post("/{dataset_id}")
async def start_training(
    dataset_id: str,
    target_column: str
):

    # Find dataset metadata

    dataset = await mongodb.database.datasets.find_one(
        {
            "_id": ObjectId(dataset_id)
        }
    )


    if not dataset:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found"
        )


    file_path = dataset["path"]

    processed_path = f"datasets/processed/processed_{dataset_id}.csv"

    result = await train_models(
        processed_path,
        dataset_id,
        target_column
    )

    return result