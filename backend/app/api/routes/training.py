from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, HTTPException, status

from app.database import mongodb
from app.services.training_services import train_models

router = APIRouter()


@router.post("/{dataset_id}", status_code=status.HTTP_200_OK)
async def start_training(dataset_id: str):
  
  try:
    obj_id = ObjectId(dataset_id)
  except InvalidId:
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Invalid dataset_id format",
    )

 
  dataset = await mongodb.database.datasets.find_one({"_id": obj_id})

  if not dataset:
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found"
    )

  
  target_column = dataset.get("target_column")
  if not target_column:
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Target column not selected for this dataset",
    )

  processed_path = f"datasets/processed/processed_{dataset_id}.csv"

  
  result = await train_models(
      file_path=processed_path,
      dataset_id=dataset_id,
      target_column=target_column,
  )

  return result