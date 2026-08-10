import app.database.mongodb as mongodb


async def save_training_result(result: dict):
    response = await mongodb.database.training_results.insert_one(result)
    return str(response.inserted_id)

async def get_training_by_dataset(dataset_id: str):

    return await mongodb.database.training_results.find_one(
        {
            "dataset_id": dataset_id
        }
    )