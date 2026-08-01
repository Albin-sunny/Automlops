import app.database.mongodb as mongodb


async def save_preprocessing(result: dict):
    response = await mongodb.database.preprocessing_results.insert_one(result)
    return str(response.inserted_id)