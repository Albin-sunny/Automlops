from fastapi import APIRouter, HTTPException, UploadFile, File

from app.services.prediction_services import predict

router = APIRouter()
from app.database import mongodb


@router.post("/{dataset_id}")
async def make_prediction(
    dataset_id: str,
    file: UploadFile = File(...)
):

    try:

        result = await predict(
            dataset_id,
            file
        )

        return result

    except FileNotFoundError:

        raise HTTPException(
            status_code=404,
            detail="Model or preprocessing artifacts not found"
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )




@router.get("/history")
async def prediction_history():

    predictions = []

    async for item in mongodb.database.predictions.find():

        item["_id"] = str(item["_id"])

        predictions.append(item)

    return predictions
    