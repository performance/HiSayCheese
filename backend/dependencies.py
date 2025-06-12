# dependencies.py
import os
import uuid
import logging
from typing import Generator

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from db import crud
from db.database import get_db
from models import models as db_models
from services.storage_service import StorageService

from services.email_service import EmailService
from fastapi import Request

logger = logging.getLogger(__name__)

# This directory will store temporary files downloaded from S3 for processing.
TEMP_PROCESSING_DIR = "/tmp/image_processing/"

# Instantiate a single StorageService instance.
# In a real-world scenario with complex DI needs, you might use a framework
# like `fastapi-injector`, but for this scale, a module-level instance is fine.
try:
    storage_service = StorageService()
    logger.info("StorageService initialized for dependencies.")
except Exception as e:
    logger.error(f"Failed to initialize StorageService in dependencies.py: {e}")
    storage_service = None # Allow app to start but fail on operations.


def get_storage_service(request: Request) -> StorageService:
    """Dependency to provide a StorageService instance initialized with the request context."""
    return StorageService(request=request)

def get_email_service(request: Request) -> EmailService:
    return EmailService(request=request)

def get_image_from_db(image_id: uuid.UUID, db: Session = Depends(get_db)) -> db_models.Image:
    """
    A dependency that retrieves an Image record from the database.
    It handles the 404 case if the image is not found.
    """
    logger.debug(f"Dependency getting image from DB for image_id: {image_id}")
    db_image = crud.get_image(db, image_id=image_id)
    if not db_image:
        logger.warning(f"Image not found in DB for id: {image_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image with id {image_id} not found."
        )
    return db_image


def get_image_local_path(
    db_image: db_models.Image = Depends(get_image_from_db)
) -> Generator[str, None, None]:
    """
    A dependency that provides a temporary local path to an image file from S3.

    This dependency is composable: it depends on `get_image_from_db` to first
    retrieve the image record, and then proceeds to download the file.

    It handles:
    - Checking for a valid S3 file path.
    - Creating a temporary directory.
    - Downloading the file from S3.
    - Yielding the temporary file path for use in the endpoint.
    - Automatically cleaning up (deleting) the temporary file afterwards.
    """
    if not storage_service:
        logger.error("StorageService not available for S3 download.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Image storage service is not configured or unavailable."
        )

    if not db_image.filepath:
        logger.warning(f"Image record for id {db_image.id} exists but has no S3 filepath.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"S3 key for image id {db_image.id} not available. Cannot process file."
        )

    temp_local_path = None
    try:
        # Ensure the temporary directory exists.
        os.makedirs(TEMP_PROCESSING_DIR, exist_ok=True)

        # Create a unique temporary file path to avoid collisions.
        base_s3_key = os.path.basename(db_image.filepath)
        temp_local_path = os.path.join(TEMP_PROCESSING_DIR, f"{uuid.uuid4().hex}_{base_s3_key}")

        logger.info(f"Downloading S3 object '{db_image.filepath}' to '{temp_local_path}' for processing.")
        storage_service.download_file(
            object_key=db_image.filepath,
            destination_path=temp_local_path
        )

        # Yield the path to the endpoint function for it to use.
        yield temp_local_path

    except HTTPException as e:
        # Re-raise HTTPExceptions (e.g., from S3 download failing) directly.
        logger.error(f"HTTPException during file preparation for image_id {db_image.id}: {e.detail}")
        raise e
    except Exception as e:
        # Catch any other unexpected errors during the process.
        logger.error(f"Unexpected error in get_image_local_path for image_id {db_image.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve and prepare image for processing."
        )
    finally:
        # This block executes after the endpoint function has finished.
        if temp_local_path and os.path.exists(temp_local_path):
            try:
                os.remove(temp_local_path)
                logger.info(f"Cleaned up temporary file: {temp_local_path}")
            except OSError as e_clean:
                # Log an error if cleanup fails, but don't crash the request.
                logger.error(f"Failed to clean up temporary file {temp_local_path}: {e_clean}")