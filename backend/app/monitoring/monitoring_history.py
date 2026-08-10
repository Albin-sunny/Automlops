# # import app.database.mongodb as mongodb


# # async def save_monitoring_result(result: dict):
# #     response = await mongodb.database.monitoring_history.insert_one(result)
# #     return str(response.inserted_id)

# import app.database.mongodb as mongodb


# async def save_monitoring_result(result: dict) -> dict:
#     response = await mongodb.database.monitoring_history.insert_one(result)
    
#     # Convert BSON ObjectId to string so FastAPI can serialize it
#     result["_id"] = str(response.inserted_id)
    
#     return result

import app.database.mongodb as mongodb


async def save_monitoring_result(result: dict) -> dict:
    response = await mongodb.database.monitoring_history.insert_one(result)
    
    # Convert BSON ObjectId to string so FastAPI can serialize it
    result["_id"] = str(response.inserted_id)
    
    return result


async def get_monitoring_history(limit: int = 20):
    cursor = (
        mongodb.database.monitoring_history
        .find({}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(limit)
    )
    results = await cursor.to_list(length=limit)

    return results


async def get_latest_monitoring_result():
    result = await (
        mongodb.database.monitoring_history
        .find_one(
            {},
            {"_id": 0},
            sort=[("timestamp", -1)]
        )
    )

    return result