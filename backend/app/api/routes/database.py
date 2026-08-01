from fastapi import APIRouter
import app.database.mongodb as mongodb

router = APIRouter()

@router.get("/")
async def database_health():
    await mongodb.database.command("ping")

    return {
        "database": "connected",
        "status": "healthy"
    }