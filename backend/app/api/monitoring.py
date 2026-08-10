from pathlib import Path
import shutil
import tempfile
from app.monitoring.monitoring_history import (
    get_monitoring_history,
    get_latest_monitoring_result,
)

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
    status,
)

from app.monitoring.monitoring_service import generate_monitoring_report

router = APIRouter(
    prefix="/monitoring",
    tags=["Monitoring"],
)

TEMP_DIR = Path("app/data/temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def _save_upload_chunked(upload_file: UploadFile, destination: Path) -> None:
    """Streams uploaded chunks directly to disk to prevent OOM errors."""
    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)


@router.post("/compare")
async def compare_datasets(
    reference_file: UploadFile = File(...),
    current_file: UploadFile = File(...),
):
    # 1. Extension Validation
    for file, label in [
        (reference_file, "Reference"),
        (current_file, "Current"),
    ]:
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{label} file must be a CSV file",
            )

    # Generate unique isolated temporary paths
    ref_path = (
        TEMP_DIR / f"ref_{tempfile.mktemp(suffix='.csv', dir='')}"
    )
    curr_path = (
        TEMP_DIR / f"curr_{tempfile.mktemp(suffix='.csv', dir='')}"
    )

    try:
        # 2. Memory-Safe Chunked Writes
        _save_upload_chunked(reference_file, ref_path)
        _save_upload_chunked(current_file, curr_path)

        # 3. Execute Async Monitoring Pipeline
        result = await generate_monitoring_report(str(ref_path), str(curr_path))
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Monitoring failed: {str(e)}",
        )
    finally:
        # 4. Automated Disk Cleanup
        for path in (ref_path, curr_path):
            if path.exists():
                path.unlink()

@router.get("/history")
async def monitoring_history(limit: int = 20):

    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 100"
        )

    results = await get_monitoring_history(limit)

    return {
        "count": len(results),
        "results": results
    }


@router.get("/latest")
async def latest_monitoring():

    result = await get_latest_monitoring_result()

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No monitoring results found"
        )

    return result