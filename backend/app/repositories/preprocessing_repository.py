import app.database.mongodb as mongodb


async def save_preprocessing(result: dict):
    response = await mongodb.database.preprocessing_results.insert_one(result)
    return str(response.inserted_id)


async def get_monitoring_history(limit: int = 20):
    cursor = (
        mongodb.database.monitoring_results
        .find({}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(limit)
    )

    results = await cursor.to_list(length=limit)

    return results


async def get_latest_monitoring_result():
    result = await (
        mongodb.database.monitoring_results
        .find_one(
            {},
            {"_id": 0},
            sort=[("timestamp", -1)]
        )
    )

    return result