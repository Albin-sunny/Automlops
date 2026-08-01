import app.database.mongodb as mongodb


async def save_training_result(result: dict):
    response = await mongodb.database.training_results.insert_one(result)
    return str(response.inserted_id)