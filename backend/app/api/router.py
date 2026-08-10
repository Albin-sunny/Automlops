from fastapi import APIRouter

from app.api.routes.health import router as health_router
from app.api.routes.database import router as database_router
from app.api.routes.dataset import router as dataset_router
from app.api.routes import profiling
from app.api.routes import preprocessing
from app.api.routes import training
from app.api.routes import prediction



router = APIRouter()

router.include_router(
    health_router,
    prefix="/health",
    tags=["Health"]
)

router.include_router(
    database_router,
    prefix="/database",
    tags=["Database"]
)

router.include_router(
    dataset_router,
    prefix="/datasets",
    tags=["Datasets"]
)

router.include_router(
    profiling.router,
    prefix="/profiling",
    tags=["Profiling"]
)


router.include_router(
    preprocessing.router,
    prefix="/preprocessing",
    tags=["Preprocessing"]
)

router.include_router(
    training.router,
    prefix="/training",
    tags=["Training"]
)

router.include_router(
    prediction.router,
    prefix="/prediction",
    tags=["Prediction"]
)