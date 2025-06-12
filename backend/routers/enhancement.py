# routers/enhancement.py
import uuid
import json
import logging
from typing import Optional
from io import BytesIO

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from PIL import Image as PILImage

from db import crud
from db.database import get_db
from models import models as db_models # For DB operations and current_user type hint
from schemas import enhancement_schemas, image_schemas, history_schemas  # Import from new schema files
from services import storage_service, image_processing, face_detection, image_quality, auto_enhancement
from dependencies import get_image_local_path, get_image_from_db # Reusable dependencies
from auth_utils import get_current_user
from rate_limiter import limiter, get_dynamic_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/enhancement",
    tags=["enhancement"],
    responses={404: {"description": "enhancement Not found"}},
)

# Instantiate a single StorageService instance to be used by this router
# In a larger app, this might be handled by a more robust DI framework.
storage = storage_service.StorageService()

# routers/enhancement.py

# ... (imports and other code) ...

# Ensure this helper context manager is defined in this file
from contextlib import asynccontextmanager
import os

@asynccontextmanager
async def get_image_local_path_from_id(image_id: uuid.UUID, db: Session):
    db_image = crud.get_image(db, image_id=image_id)
    if not db_image or not db_image.filepath:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Image {image_id} not found or has no file path.")
    
    storage = storage_service.StorageService() # Create an instance here
    TEMP_PROCESSING_DIR = "/tmp/image_processing/"
    os.makedirs(TEMP_PROCESSING_DIR, exist_ok=True)
    temp_local_path = os.path.join(TEMP_PROCESSING_DIR, f"{uuid.uuid4().hex}_{os.path.basename(db_image.filepath)}")
    
    try:
        storage.download_file(object_key=db_image.filepath, destination_path=temp_local_path)
        yield temp_local_path
    finally:
        if temp_local_path and os.path.exists(temp_local_path):
            os.remove(temp_local_path)



@router.get(
    "/auto/{image_id}",
    response_model=enhancement_schemas.AutoEnhancementResponse,
    summary="Get Auto-Calculated Enhancement Parameters"
)
@limiter.limit(get_dynamic_rate_limit)
async def get_auto_enhancement_parameters(
    request: Request,
    image_id: uuid.UUID,
    db_image: db_models.Image = Depends(get_image_from_db), # Dependency to get DB image
    temp_image_path: str = Depends(get_image_local_path), # Dependency to get local file path
    mode: Optional[str] = None
):
    """
    Calculates suggested enhancement parameters for an image based on automated analysis.

    - **image_id**: UUID of the image to analyze.
    - **mode**: Optional mode (e.g., "passport") to tailor suggestions.
    - This endpoint chains dependencies to first get the image record from the DB,
      then download the file from S3 for processing.
    """
    logger.info(f"Request for auto enhancement for image_id: {image_id}, mode: {mode}")

    if db_image.width is None or db_image.height is None:
        logger.error(f"Image dimensions missing for image_id: {image_id}.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Cannot calculate enhancements: image dimensions are missing."
        )

    try:
        # Perform analyses on the local temporary file
        face_results = face_detection.detect_faces(temp_image_path)
        quality_results = image_quality.analyze_image_quality(temp_image_path)

        # Calculate the enhancement parameters
        enhancement_params_dict = auto_enhancement.calculate_auto_enhancements(
            image_path=temp_image_path,
            image_dimensions=(db_image.width, db_image.height),
            face_detection_results=face_results,
            image_quality_results=quality_results,
            mode=mode
        )
        enhancement_params_model = enhancement_schemas.EnhancementParameters(**enhancement_params_dict)

        logger.info(f"Successfully calculated auto-enhancement parameters for image_id: {image_id}")
        return enhancement_schemas.AutoEnhancementResponse(
            image_id=image_id,
            enhancement_parameters=enhancement_params_model,
            message="Auto enhancement parameters calculated successfully."
        )
    except (ValueError, image_processing.ImageProcessingError) as e:
        logger.error(f"Error during analysis for auto-enhancement of image_id {image_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to analyze image for auto-enhancement: {e}"
        )
    except Exception as e:
        logger.error(f"Unexpected error calculating auto-enhancements for image_id {image_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while calculating enhancement parameters."
        )


# Corrected endpoint signature and logic
@router.post(
    "/apply",
    response_model=image_schemas.ProcessedImageResponse,
    summary="Apply Manual Enhancements to an Image"
)
@limiter.limit(get_dynamic_rate_limit)
async def apply_manual_enhancements(
    request: Request,
    enhancement_request: enhancement_schemas.ImageEnhancementRequest,
    db: Session = Depends(get_db),
    current_user: db_models.User = Depends(get_current_user)
    # REMOVED: temp_image_path: str = Depends(get_image_local_path) <-- THIS WAS THE ERROR
):
    image_id = enhancement_request.image_id
    logger.info(f"User {current_user.id} applying manual enhancements to image_id: {image_id}")
    
    db_image = crud.get_image(db, image_id=image_id)
    if not db_image:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Image {image_id} not found.")

    # Use the local async context manager helper
    async with get_image_local_path_from_id(image_id, db) as temp_image_path:
        # The 'try...except' block should be inside the 'with' to handle processing errors
        try:
            face_results = []
            if enhancement_request.parameters.face_smooth_intensity > 0:
                face_results = face_detection.detect_faces(temp_image_path)

            processed_pil_image = image_processing.apply_enhancements(
                image_path=temp_image_path,
                params=enhancement_request.parameters.model_dump(),
                face_detection_results=face_results
            )

            storage = storage_service.StorageService(request=request) # Pass request for URL building
            return await _process_and_store_enhanced_image(
                db=db,
                storage=storage, # Pass the storage instance
                original_image=db_image,
                processed_pil_image=processed_pil_image,
                parameters=enhancement_request.parameters.model_dump(),
                user_id=current_user.id
            )
        except image_processing.ImageProcessingError as e:
            logger.error(f"ImageProcessingError for image_id {image_id}: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
        except Exception as e:
            msg = f"Unexpected error applying enhancements to image_id {image_id}: {e}"
            logger.error(msg, exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=msg)


@router.post(
    "/apply-preset/{preset_id}",
    response_model=image_schemas.ProcessedImageResponse,
    summary="Apply a Saved Preset to an Image"
)
@limiter.limit(get_dynamic_rate_limit)
async def apply_preset_to_image(
    request: Request,
    preset_id: uuid.UUID,
    apply_request: enhancement_schemas.ApplyPresetRequest,
    temp_image_path: str = Depends(get_image_local_path), # Inject the path directly using Depends!
    db: Session = Depends(get_db),
    current_user: db_models.User = Depends(get_current_user)
):
    """
    Applies a user's saved preset to a specified image.
    """
    image_id = apply_request.image_id
    logger.info(f"User {current_user.id} applying preset {preset_id} to image {image_id}")

    # 1. Fetch and validate the user's preset
    db_preset = crud.get_user_preset(db, preset_id=preset_id, user_id=current_user.id)
    if not db_preset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preset not found or not owned by user.")

    try:
        params_dict = json.loads(db_preset.parameters_json)
        enhancement_params = enhancement_schemas.EnhancementRequestParams(**params_dict)
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Preset {preset_id} contains invalid JSON parameters: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Preset is corrupt and cannot be applied.")

    # 2. Get original image and apply enhancements (similar to manual apply)
    db_image = crud.get_image(db, image_id=image_id)

    try:
        face_results = []
        if enhancement_params.face_smooth_intensity > 0:
            face_results = face_detection.detect_faces(temp_image_path)

        processed_pil_image = image_processing.apply_enhancements(
            image_path=temp_image_path,
            params=enhancement_params.model_dump(),
            face_detection_results=face_results
        )

        # Add preset ID to history for tracking
        history_params = enhancement_params.model_dump()
        history_params["applied_preset_id"] = str(preset_id)

        return await _process_and_store_enhanced_image(
            db=db,
            original_image=db_image,
            processed_pil_image=processed_pil_image,
            parameters=history_params, # Pass the dict with the preset ID
            user_id=current_user.id
        )

    except image_processing.ImageProcessingError as e:
        logger.error(f"ImageProcessingError for image_id {image_id} with preset {preset_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        msg = f"Unexpected error applying preset {preset_id} to image {image_id}: {e}"
        logger.error(msg, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=msg)
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)

async def _process_and_store_enhanced_image(
    db: Session,
    storage: storage_service.StorageService,
    original_image: db_models.Image,
    processed_pil_image: PILImage.Image,
    parameters: dict, # Can be EnhancementRequestParams or dict with preset info
    user_id: uuid.UUID
) -> image_schemas.ProcessedImageResponse:
    """
    Helper function to handle the common logic of:
    - Saving a processed PIL image to a BytesIO stream.
    - Uploading it to S3.
    - Creating the new Image DB record.
    - Creating the EnhancementHistory DB record.
    - Generating a presigned URL for the response.
    """
    if not processed_pil_image:
        raise image_processing.ImageProcessingError("Image processing resulted in a null image.")

    # Convert processed PIL image to bytes in memory
    image_bytes_io = BytesIO()
    processed_pil_image.save(image_bytes_io, format='PNG') # Standardize on PNG for processed images
    image_bytes_io.seek(0)

    # Upload to S3
    s3_key = f"processed_images/{original_image.id}_enhanced_{uuid.uuid4().hex}.png"
    storage.upload_file(
        file_obj=image_bytes_io,
        object_key=s3_key,
        content_type='image/png'
    )
    logger.info(f"Uploaded enhanced image to S3 key: {s3_key}")

    # Create DB record for the new processed image
    processed_image_record = crud.create_image(db, image_schemas.ImageCreate(
        filename=os.path.basename(s3_key),
        filepath=s3_key,
        filesize=image_bytes_io.getbuffer().nbytes,
        mimetype='image/png',
        width=processed_pil_image.width,
        height=processed_pil_image.height,
        format=processed_pil_image.format
    ))

    # Create enhancement history record
    crud.create_enhancement_history(db, history_schemas.EnhancementHistoryBase(
        original_image_id=original_image.id,
        processed_image_id=processed_image_record.id,
        parameters_json=json.dumps(parameters if isinstance(parameters, dict) else parameters.model_dump())
    ), user_id=user_id)

    # Generate a presigned URL for client access
    presigned_url = storage.generate_presigned_url(s3_key)

    return image_schemas.ProcessedImageResponse(
        original_image_id=original_image.id,
        processed_image_id=processed_image_record.id,
        processed_image_path=presigned_url,
        message="Image enhanced successfully."
    )
