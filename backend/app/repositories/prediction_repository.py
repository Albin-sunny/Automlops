from app.database import mongodb


async def save_prediction(data: dict):

    result = await mongodb.database.predictions.insert_one(data)

    return str(result.inserted_id)