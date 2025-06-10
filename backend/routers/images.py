import os
import tempfile
import uuid # Needed for generating unique object keys
import cv2 # Needed for imencode
import io # Needed for BytesIO
from typing import List, Optional, Dict, Any # For type hinting

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.services.storage_service import StorageService
from backend.services.face_detection import detect_faces
from backend.services.image_quality import analyze_image_quality
from backend.services.apply_image_modifications import apply_all_enhancements
# calculate_auto_enhancements is not used in this version as per instructions
# from backend.services.auto_enhancement import calculate_auto_enhancements

from backend.auth_utils import get_current_user
from backend.models.models import User as UserModel # For type hinting current_user

# --- Pydantic Models ---

class AnalyzeImageRequest(BaseModel):
    object_key: str

class EnhancementParametersRequest(BaseModel):
    brightness_target: Optional[float] = Field(default=0.0)
    contrast_target: Optional[float] = Field(default=1.0)
    saturation_target: Optional[float] = Field(default=1.0)
    background_blur_radius: Optional[int] = Field(default=0)
    crop_rect: Optional[List[int]] = Field(default=None, example=[0,0,100,100]) # x,y,w,h
    face_smooth_intensity: Optional[float] = Field(default=0.0)
    # face_boxes are not part of the request directly, they are detected if needed

class ApplyEnhancementsRequest(BaseModel):
    original_object_key: str
    enhancements: EnhancementParametersRequest
    mode: Optional[str] = Field(default="default") # For future use or to influence defaults

class ApplyEnhancementsResponse(BaseModel):
    original_object_key: str
    enhanced_object_key: str
    enhanced_file_url: str

# --- Router Definition ---

router = APIRouter(
    prefix="/api/images",
    tags=["images"],
    responses={
        404: {"description": "Image not found in S3 or analysis/processing failed for specific reasons"},
        400: {"description": "Bad request, e.g., unprocessable image or invalid parameters"},
        401: {"description": "Unauthorized"},
        422: {"description": "Validation error in request body"},
    }
)

# --- Endpoints ---

@router.post("/analyze")
async def analyze_image_endpoint(
    request_data: AnalyzeImageRequest,
    current_user: UserModel = Depends(get_current_user),
    storage_service: StorageService = Depends(StorageService)
):
    """
    Analyzes an image from S3 for face detection and image quality.
    """
    temp_file_path: Optional[str] = None
    try:
        _, file_extension = os.path.splitext(request_data.object_key)
        if not file_extension:
            file_extension = ".tmp"

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            temp_file_path = tmp_file.name

        storage_service.download_file(
            object_key=request_data.object_key,
            destination_path=temp_file_path
        )

        face_results = detect_faces(image_path=temp_file_path)
        quality_results = analyze_image_quality(image_path=temp_file_path)

        return {
            "object_key": request_data.object_key,
            "user_id": current_user.id,
            "face_detection": face_results,
            "image_quality": quality_results
        }
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Temporary image file not found after download.")
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Image format error or unprocessable image: {str(ve)}")
    except HTTPException:
        raise
    except Exception as e:
        # logger.error(f"Unexpected error during image analysis for {request_data.object_key}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred during image analysis: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception: # pragma: no cover
                # logger.error(f"Failed to remove temporary file {temp_file_path}: {e_remove}", exc_info=True)
                pass


@router.post("/apply-enhancements", response_model=ApplyEnhancementsResponse)
async def apply_enhancements_endpoint(
    request_data: ApplyEnhancementsRequest,
    current_user: UserModel = Depends(get_current_user),
    storage_service: StorageService = Depends(StorageService)
):
    """
    Downloads an image from S3, applies specified enhancements,
    uploads the modified image back to S3, and returns its new key and URL.
    """
    temp_original_image_path: Optional[str] = None
    try:
        # Determine file extension for temporary file and output
        _, original_file_extension = os.path.splitext(request_data.original_object_key)
        if not original_file_extension: # Default if no extension
            original_file_extension = ".jpg" # Assume JPEG if original has no extension

        output_content_type = "image/jpeg" # Default output
        if original_file_extension.lower() == ".png":
            output_content_type = "image/png"
        # Add more types like webp if necessary

        # Create a temporary file for the original image
        with tempfile.NamedTemporaryFile(delete=False, suffix=original_file_extension) as tmp_file:
            temp_original_image_path = tmp_file.name

        # Download original image
        storage_service.download_file(
            object_key=request_data.original_object_key,
            destination_path=temp_original_image_path
        )

        # Prepare enhancement parameters
        enhancement_params_dict = request_data.enhancements.model_dump()

        # If face smoothing is requested, detect faces and add to params
        if enhancement_params_dict.get("face_smooth_intensity", 0.0) > 0.0:
            # logger.info(f"Face smoothing requested for {request_data.original_object_key}, detecting faces.")
            face_detection_results = detect_faces(image_path=temp_original_image_path)
            enhancement_params_dict["face_boxes"] = [f["box"] for f in face_detection_results.get("faces", []) if f.get("box")]


        # Apply enhancements
        modified_image_np = apply_all_enhancements(
            image_path=temp_original_image_path,
            enhancement_params=enhancement_params_dict
        )

        if modified_image_np is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Image processing failed, possibly due to an unsupported image format or an error during enhancement.")

        # Encode modified image to bytes
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90] if output_content_type == "image/jpeg" else None
        status_encode, image_bytes_np = cv2.imencode(original_file_extension, modified_image_np, encode_param)
        if not status_encode:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to encode modified image.")

        image_bytes = image_bytes_np.tobytes()

        # Create new object key for the enhanced image
        new_object_key = f"enhanced_images/{current_user.id}/{uuid.uuid4()}{original_file_extension}"

        # Upload enhanced image to S3
        storage_service.upload_file(
            file_obj=io.BytesIO(image_bytes),
            object_key=new_object_key,
            content_type=output_content_type # e.g., 'image/jpeg' or 'image/png'
        )

        enhanced_file_url = storage_service.get_public_url(new_object_key)

        return ApplyEnhancementsResponse(
            original_object_key=request_data.original_object_key,
            enhanced_object_key=new_object_key,
            enhanced_file_url=enhanced_file_url
        )

    except FileNotFoundError: # Should be caught by storage_service.download_file as 404
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Temporary image file not found after download (apply-enhancements).")
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Image format error or unprocessable image (apply-enhancements): {str(ve)}")
    except HTTPException: # Re-raise HTTPExceptions (e.g. from download_file or if processing failed with specific HTTP status)
        raise
    except Exception as e:
        # logger.error(f"Unexpected error during image enhancement for {request_data.original_object_key}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred during image enhancement: {str(e)}")
    finally:
        if temp_original_image_path and os.path.exists(temp_original_image_path):
            try:
                os.remove(temp_original_image_path)
            except Exception: # pragma: no cover
                # logger.error(f"Failed to remove temporary original image file {temp_original_image_path}: {e_remove}", exc_info=True)
                pass
