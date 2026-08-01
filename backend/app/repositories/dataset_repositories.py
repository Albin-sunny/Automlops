import app.database.mongodb as mongodb
from bson import ObjectId

async def save_dataset(metadata: dict):
    result = await mongodb.database.datasets.insert_one(metadata.copy())
    return str(result.inserted_id)




async def get_all_datasets():
    datasets = []

    cursor = mongodb.database.datasets.find()

    async for dataset in cursor:
        dataset["_id"] = str(dataset["_id"])
        datasets.append(dataset)

    return datasets

async def get_dataset_by_id(dataset_id: str):

    from bson import ObjectId

    dataset = await mongodb.database.datasets.find_one(
        {"_id": ObjectId(dataset_id)}
    )

    if dataset:
        dataset["_id"] = str(dataset["_id"])

    return dataset


async def delete_dataset(dataset_id: str):

    result = await mongodb.database.datasets.delete_one(
        {"_id": ObjectId(dataset_id)}
    )

    return result.deleted_count