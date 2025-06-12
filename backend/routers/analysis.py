# routers/analysis.py
import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session

from db import crud
from db.database import get_db
from schemas import analysis_schemas  # Import response models from the new schemas directory
from services import face_detection, image_quality
from dependencies import get_image_local_path  # Import the new reusable dependency
from rate_limiter import limiter, get_dynamic_rate_limit

# Initialize logger for this module
logger = logging.getLogger(__name__)

##############


router = APIRouter(
    prefix="/api/analysis",
    tags=["Image Analysis"],
    responses={404: {"description": "Image Analysis Not found"}},
)


@router.get(
    "/faces/{image_id}",
    response_model=analysis_schemas.FaceDetectionResponse,
    summary="Detect Faces in an Image"
)
@limiter.limit(get_dynamic_rate_limit)
async def get_face_detections(
    request: Request,
    image_id: uuid.UUID,
    temp_image_path: str = Depends(get_image_local_path)
):
    """
    Analyzes an uploaded image to detect human faces and returns their bounding boxes.

    - **image_id**: The UUID of the image to analyze.
    - This endpoint uses a dependency to download the image from S3 to a temporary
      local path for processing and handles cleanup automatically.
    """
    logger.info(f"Request to detect faces for image_id: {image_id}")

    try:
        # The dependency has already handled DB lookup, S3 download, and temp file creation.
        # We just need to call the face detection service with the provided path.
        detected_faces_data = face_detection.detect_faces(temp_image_path)

        # Format the data into the Pydantic response model.
        response_faces = [
            analysis_schemas.FaceBoundingBox(box=f['box'], confidence=f.get('confidence'))
            for f in detected_faces_data
        ]

        if not response_faces:
            logger.info(f"No faces detected for image_id: {image_id}")
            return analysis_schemas.FaceDetectionResponse(
                faces=[],
                image_id=image_id,
                message="No faces detected."
            )

        logger.info(f"Successfully detected {len(response_faces)} faces for image_id: {image_id}")
        return analysis_schemas.FaceDetectionResponse(faces=response_faces, image_id=image_id)

    except ValueError as e:
        # This error comes from the face_detection service if the image is invalid.
        logger.error(f"Invalid image file for face detection (image_id: {image_id}): {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid or corrupt image file: {str(e)}"
        )
    except Exception as e:
        # Catch any other unexpected errors during the analysis.
        logger.error(f"Unexpected error during face detection for image_id {image_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during face detection."
        )


@router.get(
    "/quality/{image_id}",
    response_model=analysis_schemas.ImageQualityAnalysisResponse,
    summary="Analyze Image Quality Metrics"
)
@limiter.limit(get_dynamic_rate_limit)
async def get_image_quality_analysis(
    request: Request,
    image_id: uuid.UUID,
    temp_image_path: str = Depends(get_image_local_path)
):
    """
    Analyzes an uploaded image for quality metrics like brightness and contrast.

    - **image_id**: The UUID of the image to analyze.
    - Utilizes the `get_image_local_path` dependency for file handling.
    """
    logger.info(f"Request to analyze quality for image_id: {image_id}")

    try:
        # Call the image quality service. The dependency handles all file prep.
        quality_data = image_quality.analyze_image_quality(temp_image_path)

        # Format the data into the Pydantic response models.
        metrics = analysis_schemas.ImageQualityMetrics(
            brightness=quality_data["brightness"],
            contrast=quality_data["contrast"]
        )

        logger.info(f"Successfully analyzed image quality for image_id: {image_id}")
        return analysis_schemas.ImageQualityAnalysisResponse(
            image_id=image_id,
            quality_metrics=metrics,
            insights=quality_data["insights"],
            message="Image quality analysis successful."
        )

    except ValueError as e:
        # This error comes from the image_quality service if the image is invalid.
        logger.error(f"Invalid image file for quality analysis (image_id: {image_id}): {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid or corrupt image file: {str(e)}"
        )
    except Exception as e:
        # Catch any other unexpected errors during the analysis.
        logger.error(f"Unexpected error during image quality analysis for image_id {image_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during image quality analysis."
        )