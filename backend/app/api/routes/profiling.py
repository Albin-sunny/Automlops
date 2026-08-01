from fastapi import APIRouter, HTTPException
from bson import ObjectId

from app.database import mongodb
from app.services.profiling_services import generate_profile

router = APIRouter()


@router.post("/{dataset_id}")
async def create_profile(dataset_id: str):

    dataset = await mongodb.database.datasets.find_one(
        {"_id": ObjectId(dataset_id)}
    )

    if not dataset:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found"
        )

    file_path = dataset["path"]

    profile = await generate_profile(
        file_path,
        dataset_id
    )

    return profile