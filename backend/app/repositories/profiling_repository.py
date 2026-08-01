import app.database.mongodb as mongodb


async def save_profile(profile_data: dict):

    result = await mongodb.database.profiling_results.insert_one(
        profile_data.copy()
    )

    return str(result.inserted_id)