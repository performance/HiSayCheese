# routers/images.py
import uuid
import logging
import magic
import os
from io import BytesIO

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, Request
from sqlalchemy.orm import Session
from werkzeug.utils import secure_filename
from PIL import Image as PILImage

from db import crud
from db.database import get_db
from models import models as db_models
from schemas import image_schemas, analysis_schemas
from services import storage_service
from services.content_moderation import moderate_image_content
from rate_limiter import limiter, get_dynamic_rate_limit
from pydantic import BaseModel

from dependencies import get_storage_service
from services.storage_service import StorageService

logger = logging.getLogger(__name__)

# THIS IS THE KEY CHANGE: We define a router here.
router = APIRouter(
    prefix="/api/images",
    tags=["Image Upload"]
)

# Constants moved here as they are specific to this router's logic.
ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"]
MIME_TYPE_TO_EXTENSION = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
MAX_FILE_SIZE_MB = 15
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# Placeholder for a more robust malware scanning service
async def scan_for_malware(contents: bytes) -> bool:
    """Placeholder for malware scanning logic."""
    logger.info("Malware scan stub: assuming file is safe.")
    return True


# THE FIX IS HERE: The decorator now uses `@router.post` instead of `@app.post`
@router.post(
    "/upload",
    response_model=image_schemas.ImageSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Upload, Validate, and Store a New Image"
)
@limiter.limit(get_dynamic_rate_limit)
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    storage: StorageService = Depends(get_storage_service)
):
    """
    Handles the full image upload process:
    1.  Validates file size and MIME type.
    2.  Performs a (placeholder) malware scan.
    3.  Moderates content using Google Cloud Vision API for portraits and safety.
    4.  Extracts image metadata.
    5.  Uploads approved images to S3.
    6.  Creates a corresponding record in the database.
    - Rejected images have their metadata saved for logging/review but are not stored in S3.
    """
    contents = await file.read()
    file_size = len(contents)

    # 1. File Size Validation
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB."
        )

    # 2. File Type Validation (using python-magic for robustness)
    detected_mime_type = magic.from_buffer(contents, mime=True)
    if detected_mime_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: '{detected_mime_type}'. Allowed types are JPG, PNG, WEBP."
        )
    file_extension = MIME_TYPE_TO_EXTENSION.get(detected_mime_type)

    # 3. Malware Scan
    if not await scan_for_malware(contents):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Malware detected in file."
        )

    # 4. Content Moderation
    moderation_result = await moderate_image_content(contents)

    # 5. Extract Metadata
    sanitized_filename = secure_filename(file.filename) or f"upload_{uuid.uuid4().hex}"
    try:
        image_pil = PILImage.open(BytesIO(contents))
        width, height = image_pil.size
        img_format = image_pil.format
        exif = image_pil.getexif()
        exif_orientation = exif.get(0x0112) if exif else None
        color_profile = "ICC" if 'icc_profile' in image_pil.info else image_pil.mode
    except Exception as e:
        logger.error(f"Could not extract metadata from image file '{sanitized_filename}': {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Cannot process image file. It may be corrupt.")

    # 6. Create initial image data for DB, handle rejection case
    image_data = image_schemas.ImageCreate(
        filename=sanitized_filename,
        filepath=None,  # Set later if approved
        filesize=file_size,
        mimetype=detected_mime_type,
        width=width,
        height=height,
        format=img_format,
        exif_orientation=exif_orientation,
        color_profile=color_profile,
        rejection_reason=moderation_result.rejection_reason
    )

    if not moderation_result.is_approved:
        logger.info(f"Image '{sanitized_filename}' rejected: {moderation_result.rejection_reason}")
        crud.create_image(db=db, image=image_data) # Log rejected image metadata
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=moderation_result.rejection_reason
        )

    # 7. Approved: Upload to S3 and finalize DB record
    s3_object_key = f"original_images/{uuid.uuid4().hex}{file_extension}"
    try:
        storage.upload_file(
            file_obj=BytesIO(contents),
            object_key=s3_object_key,
            content_type=detected_mime_type
        )
        logger.info(f"Approved image '{sanitized_filename}' uploaded to S3 with key: {s3_object_key}")
        image_data.filepath = s3_object_key # Update filepath with S3 key
        image_data.rejection_reason = None # Ensure rejection reason is null for approved images
    except Exception as e:
        logger.error(f"S3 upload failed for '{sanitized_filename}': {e}", exc_info=True)
        # Attempt to save a record indicating upload failure
        image_data.rejection_reason = "S3_UPLOAD_FAILED"
        crud.create_image(db=db, image=image_data)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not upload file to storage."
        )

    # Create the final DB record for the approved and uploaded image
    db_image = crud.create_image(db=db, image=image_data)

    # Generate a presigned URL for the client to immediately access the image
    presigned_url = storage.generate_presigned_url(db_image.filepath)

    # Manually construct the response to include the presigned_url, as it's not part of the DB model
    response_data = image_schemas.ImageSchema.from_orm(db_image).model_dump()
    response_data['presigned_url'] = presigned_url
    
    return response_data