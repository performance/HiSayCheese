from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging

# Assuming get_db is correctly located here based on project structure
from db.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/health",
    tags=["health"]
)

@router.get(
    "/live",
    summary="Liveness Probe",
    description="Checks if the application instance is running. Returns HTTP 200 if alive.",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "content": {"application/json": {"example": {"status": "alive"}}},
            "description": "Application is alive and running."
        }
    }
)
async def liveness_check():
    """
    Liveness probe endpoint.
    Returns HTTP 200 status code with a JSON body `{"status": "alive"}`
    if the service is running.
    """
    logger.info("Liveness check successful for /api/health/live.")
    return {"status": "alive"}

@router.get(
    "/ready",
    summary="Readiness Probe",
    description="Checks if the application is ready to serve requests, including database connectivity.",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "content": {"application/json": {"example": {"status": "ready", "detail": "Database connection successful."}}},
            "description": "Application is ready and all dependencies are connected."
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "content": {"application/json": {"example": {"status": "database_error", "detail": "Cannot connect to database."}}},
            "description": "Service unavailable, typically due to a database connection issue."
        }
    }
)
async def readiness_check(db: Session = Depends(get_db)):
    """
    Readiness probe endpoint.
    Checks dependencies like database connectivity.
    - Returns HTTP 200 if all OK.
    - Returns HTTP 503 if the database is not reachable.
    """
    try:
        # Attempt to execute a simple query to verify database connection
        db.execute(text("SELECT 1")).fetchone()
        logger.info("Readiness check: Database connection successful for /api/health/ready.")
        return {"status": "ready", "detail": "Database connection successful."}
    except Exception as e:
        logger.error(f"Readiness check: Database connection failed for /api/health/ready. Error: {e}", exc_info=True)
        # Raise HTTPException with a dictionary as detail. FastAPI will serialize this dict as the JSON response body.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "database_error", "detail": "Cannot connect to database."}
        )
