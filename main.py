# main.py
import uuid
import magic
import os # Added os import
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status # Added status
from sqlalchemy.orm import Session
from typing import Optional # Added Optional
from pydantic import BaseModel # Added BaseModel import
import os # To get environment variables
from google.cloud import vision
# from google.oauth2 import service_account # For local testing with service account key, if needed
from PIL import Image as PILImage
import io

# JSONResponse is not strictly needed if returning Pydantic model with status_code
# from fastapi.responses import JSONResponse

from db.database import create_db_and_tables, get_db
from db import crud
from models import models # This imports the models module
# We need ImageCreate and ImageSchema for type hinting and response_model
# from models.models import ImageCreate, ImageSchema # More specific imports
import logging # Added logging

# Pydantic model for content moderation result
class ContentModerationResult(BaseModel): # Corrected to BaseModel
    is_approved: bool
    rejection_reason: Optional[str] = None

app = FastAPI()

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
UPLOAD_DIR = "uploads/images/"
ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"]
MIME_TYPE_TO_EXTENSION = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
MAX_FILE_SIZE_MB = 15
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.get("/health", response_model=models.NumberSchema) # Corrected to NumberSchema
async def health(db: Session = Depends(get_db)):
    db_number = crud.get_number(db)
    if db_number is None:
        return JSONResponse(status_code=404, content={"message": "No number set yet"})
    return db_number

@app.post("/put_number", response_model=models.NumberSchema) # Corrected to NumberSchema
async def put_number(number: models.NumberCreate, db: Session = Depends(get_db)):
    return crud.create_or_update_number(db=db, value=number.value)

@app.post("/increment_number", response_model=models.NumberSchema) # Corrected to NumberSchema
async def increment_number_endpoint(db: Session = Depends(get_db)):
    db_number = crud.increment_number(db)
    if db_number is None:
        raise HTTPException(status_code=404, detail="No number set to increment.")
    return db_number

@app.get("/")
async def root():
    return {"message": "Hello World"}

# Placeholder malware scan function
async def scan_for_malware(contents: bytes) -> bool:
    """
    Placeholder for malware scanning logic.
    In a real application, this would involve integrating with a malware scanning engine.
    """
    logger.info("Malware scan stub: assuming file is safe.")
    # Simulate some async work if needed, e.g., await asyncio.sleep(0.01)
    return True

# Content moderation function using Google Cloud Vision API
async def moderate_image_content(contents: bytes) -> ContentModerationResult:
    """
    Moderates image content using Google Cloud Vision API.
    - Checks for portraits using face detection.
    - Checks for prohibited content using safe search detection.
    """
    try:
        # TODO: Configure credentials securely. For local dev, you might use GOOGLE_APPLICATION_CREDENTIALS env var.
        # Example: credentials = service_account.Credentials.from_service_account_file(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
        # client = vision.ImageAnnotatorClient(credentials=credentials)
        client = vision.ImageAnnotatorClient() # Assumes GOOGLE_APPLICATION_CREDENTIALS is set in the environment

        # Get Image Dimensions using Pillow
        try:
            image_pil = PILImage.open(io.BytesIO(contents))
            img_width, img_height = image_pil.size
            logger.info(f"Image dimensions: {img_width}x{img_height}")
        except Exception as e:
            logger.error(f"Could not open image with Pillow to get dimensions: {e}")
            return ContentModerationResult(is_approved=False, rejection_reason="Failed to process image properties.")

        image = vision.Image(content=contents)

        # Perform Face Detection
        logger.info("Performing face detection via Vision API.")
        face_response = client.face_detection(image=image)
        faces = face_response.face_annotations

        is_portrait = False
        portrait_rejection_reason = "No clear portrait found or face is not prominent." # Updated reason
        MIN_FACE_TO_IMAGE_AREA_RATIO = 0.05 # 5% - configurable

        if faces:
            image_area = img_width * img_height
            if image_area == 0: # Avoid division by zero if image dimensions are invalid
                 logger.error("Image area is zero, cannot calculate face ratio.")
                 return ContentModerationResult(is_approved=False, rejection_reason="Invalid image dimensions.")

            for face in faces:
                # Get bounding box vertices
                vertices = face.bounding_poly.vertices
                # Calculate face bounding box area
                # Ensure vertices exist and are as expected (at least 2 for min/max to be meaningful)
                if len(vertices) < 2: continue # Should not happen with valid Vision API response

                face_width = max(v.x for v in vertices) - min(v.x for v in vertices)
                face_height = max(v.y for v in vertices) - min(v.y for v in vertices)
                face_area = face_width * face_height

                # Check detection confidence and area ratio
                if face.detection_confidence > 0.75 and (face_area / image_area) >= MIN_FACE_TO_IMAGE_AREA_RATIO:
                    is_portrait = True
                    portrait_rejection_reason = None
                    logger.info(f"Portrait detected: Face area {face_area}, Image area {image_area}, Ratio {face_area/image_area:.2f}, Confidence {face.detection_confidence:.2f}")
                    break # Found a suitable portrait face
            if not is_portrait: # Log if faces were found but none met criteria
                 logger.info("Faces detected, but none met the prominence or confidence criteria for a portrait.")
        else:
            logger.info("No faces detected by Vision API.")

        if face_response.error.message:
            logger.error(f'Face detection error from Vision API: {face_response.error.message}')
            # If API call itself had an error, consider it a moderation failure.
            return ContentModerationResult(is_approved=False, rejection_reason=f"Face detection failed: {face_response.error.message}")

        # If not a portrait, reject immediately based on policy
        if not is_portrait:
            logger.info(f"Image rejected: {portrait_rejection_reason}")
            return ContentModerationResult(is_approved=False, rejection_reason=portrait_rejection_reason)

        # Perform SafeSearch Detection (only if it's a portrait)
        logger.info("Performing safe search detection via Vision API for portrait.")
        safe_search_response = client.safe_search_detection(image=image)
        detection = safe_search_response.safe_search_annotation

        prohibited_content_detected = False
        prohibited_rejection_reason = None

        PROHIBITED_LEVELS = [vision.Likelihood.LIKELY, vision.Likelihood.VERY_LIKELY]

        if (detection.adult in PROHIBITED_LEVELS or
            detection.violence in PROHIBITED_LEVELS or
            detection.racy in PROHIBITED_LEVELS): # Add other categories like medical, spoof if needed
            prohibited_content_detected = True
            prohibited_rejection_reason = "Image content violates guidelines (adult, violence, or racy content detected)."
            logger.info(f"Image rejected due to prohibited content: {prohibited_rejection_reason}")
        else:
            logger.info("Safe search: No prohibited content detected at LIKELY or VERY_LIKELY levels.")

        if safe_search_response.error.message:
            logger.error(f'SafeSearch error from Vision API: {safe_search_response.error.message}')
            # Decide if this is a hard fail
            return ContentModerationResult(is_approved=False, rejection_reason=f"SafeSearch detection failed: {safe_search_response.error.message}")

        if prohibited_content_detected:
            return ContentModerationResult(is_approved=False, rejection_reason=prohibited_rejection_reason)

        logger.info("Image approved by content moderation.")
        return ContentModerationResult(is_approved=True, rejection_reason=None)

    except Exception as e:
        logger.error(f"Error during Google Cloud Vision API call: {e}")
        return ContentModerationResult(is_approved=False, rejection_reason="Content moderation check failed due to an internal error.")

# Updated image upload endpoint
@app.post("/api/images/upload", response_model=models.ImageSchema, status_code=status.HTTP_201_CREATED)
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = await file.read()
    file_size = len(contents)

    # 1. File Size Validation
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413, # Payload Too Large
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB."
        )

    # 2. File Type Validation
    # Initial check based on browser-provided content type (file.content_type)
    # More robust check using python-magic
    detected_mime_type = magic.from_buffer(contents, mime=True)

    if detected_mime_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: '{detected_mime_type}'. Allowed types are JPG, PNG, WEBP."
        )

    actual_content_type = detected_mime_type
    file_extension = MIME_TYPE_TO_EXTENSION.get(actual_content_type)

    if not file_extension:
        # This should ideally not happen if ALLOWED_MIME_TYPES and MIME_TYPE_TO_EXTENSION are synced
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File type processed correctly, but no extension mapping found."
        )

    # 3. Malware Scan (Placeholder)
    # This is called after initial validation but before saving to disk or DB
    if not await scan_for_malware(contents):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Malware detected in file."
        )

    # 4. Content Moderation
    moderation_result = await moderate_image_content(contents)

    # Create upload directory if it doesn't exist (only if approved, or always if storing rejected metadata)
    # For now, let's assume we might want to log/store info about rejected files without saving them,
    # or save them to a different location. The current setup saves the file first, then decides.
    # Let's adjust to create ImageCreate with moderation result first.
    # os.makedirs(UPLOAD_DIR, exist_ok=True) # Moved down

    # Initial ImageCreate data, filepath might be updated if approved
    # Extract metadata using Pillow
    img_width, img_height, img_format, exif_orientation, color_profile = None, None, None, None, None
    try:
        image_pil = PILImage.open(io.BytesIO(contents))
        img_width, img_height = image_pil.size
        img_format = image_pil.format

        # EXIF Orientation
        exif_data = image_pil._getexif()
        if exif_data:
            exif_orientation = exif_data.get(0x0112) # EXIF Orientation tag

        # Color Profile
        if image_pil.info.get('icc_profile'):
            color_profile = "ICC"
        else:
            color_profile = image_pil.mode # e.g., 'RGB', 'RGBA', 'L'

    except Exception as e:
        logger.error(f"Error extracting metadata with Pillow: {e}")
        # Decide if this is fatal or if we proceed with None for metadata
        # For now, proceed with None, as these fields are optional

    image_data_to_create = models.ImageCreate(
        filename=file.filename,
        filepath=None, # Default to None, set if image is approved and saved
        filesize=file_size,
        mimetype=actual_content_type,
        width=img_width,
        height=img_height,
        format=img_format,
        exif_orientation=exif_orientation,
        color_profile=color_profile,
        rejection_reason=moderation_result.rejection_reason
    )

    if not moderation_result.is_approved:
        logger.info(f"Image rejected: {moderation_result.rejection_reason}. Original filename: {file.filename}")
        # Save metadata for rejected image, filepath is None as file is not saved.
        db_image = crud.create_image(
            db=db,
            image=image_data_to_create,
            width=image_data_to_create.width,
            height=image_data_to_create.height,
            format=image_data_to_create.format,
            exif_orientation=image_data_to_create.exif_orientation,
            color_profile=image_data_to_create.color_profile
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=moderation_result.rejection_reason
        )
    else:
        # Approved: Ensure UPLOAD_DIR exists, generate filename, save file, update filepath
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        unique_filename_base = uuid.uuid4().hex
        unique_filename = f"{unique_filename_base}{file_extension}"
        server_filepath = os.path.join(UPLOAD_DIR, unique_filename)

        try:
            with open(server_filepath, "wb") as f:
                f.write(contents)
            logger.info(f"Approved image saved to {server_filepath}")
            image_data_to_create.filepath = server_filepath # Update filepath for approved image
        except IOError as e:
            logger.error(f"Could not save approved file to {server_filepath}: {e}")
            # This is a server error after approval.
            # The image record will NOT be created in the DB if saving fails.
            # This is different from previous: if create_image was called before saving file.
            # Current: moderate -> [if approved] save file -> create_image record.
            # This seems more robust.
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not save file after approval: {e}",
            )

        # Ensure rejection_reason is explicitly None for approved images in the database
        image_data_to_create.rejection_reason = None # This is already part of image_data_to_create

        # Pass all required fields to crud.create_image
        db_image = crud.create_image(
            db=db,
            image=image_data_to_create, # contains filename, filepath, filesize, mimetype, rejection_reason
            width=image_data_to_create.width,
            height=image_data_to_create.height,
            format=image_data_to_create.format,
            exif_orientation=image_data_to_create.exif_orientation,
            color_profile=image_data_to_create.color_profile
        )
        return db_image
