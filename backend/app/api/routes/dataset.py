from fastapi import APIRouter, UploadFile, File, HTTPException
from bson import ObjectId

from app.database import mongodb
from app.services.dataset_services import process_dataset
from app.services.dataset_services import fetch_all_datasets
from app.services.dataset_services import fetch_dataset_by_id
from app.services.dataset_services import remove_dataset

router = APIRouter()


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    return await process_dataset(file)


@router.get("/")
async def get_datasets():

    return await fetch_all_datasets()


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str):

    return await fetch_dataset_by_id(dataset_id)


@router.delete("/{dataset_id}")
async def delete_dataset_route(dataset_id: str):

    return await remove_dataset(dataset_id)


@router.post("/{dataset_id}/select-target")
async def select_target(
    dataset_id: str,
    target_column: str
):

    result = await mongodb.database.datasets.update_one(
        {
            "_id": ObjectId(dataset_id)
        },
        {
            "$set": {
                "target_column": target_column
            }
        }
    )

    if result.matched_count == 0:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found"
        )

    return {
        "message": "Target column selected successfully",
        "dataset_id": dataset_id,
        "target_column": target_column
    }