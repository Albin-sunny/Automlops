from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import router
from app.api.monitoring import router as monitoring_router
# from app.api.agent import router as agent_router

from app.database.mongodb import (
    connect_to_mongo,
    close_mongo_connection
)


@asynccontextmanager
async def lifespan(app: FastAPI):

    await connect_to_mongo()

    yield

    await close_mongo_connection()


app = FastAPI(
    title="AutoMLOps API",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router)
app.include_router(monitoring_router)


@app.get("/")
async def root():
    return {
        "message": "Welcome to AutoMLOps API"
    }


