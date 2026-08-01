from motor.motor_asyncio import AsyncIOMotorClient

client: AsyncIOMotorClient | None = None
database = None


async def connect_to_mongo():
    global client, database

    client = AsyncIOMotorClient("mongodb://localhost:27017")

    database = client["automlops"]

    print("✅ Connected to MongoDB")


async def close_mongo_connection():
    global client

    if client:
        client.close()
        print("❌ MongoDB Connection Closed")