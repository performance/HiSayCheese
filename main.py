# main.py
import uuid
import magic
import os # Added os import
from werkzeug.utils import secure_filename
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status # Added status
import json # Added for EnhancementHistory
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
from fastapi.middleware.cors import CORSMiddleware # Import for CORS
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseCall
from starlette.requests import Request
from starlette.responses import Response as StarletteResponse # Renamed to avoid conflict with FastAPI's Response
from starlette.types import ASGIApp # Added for middleware

# Rate Limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.models import Limit # For parsing rate limit strings
from slowapi.util import parse_many # To parse rate limit strings like "100/minute"

from fastapi.responses import JSONResponse # For the exception handler
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from config import SECRET_KEY, ALGORITHM # For decoding JWT

from db.database import create_db_and_tables, get_db
from db import crud
from models import models # This imports the models module
# We need ImageCreate and ImageSchema for type hinting and response_model
# from models.models import ImageCreate, ImageSchema # More specific imports
# Corrected import for models to include User, EnhancementHistoryBase, ImageCreate
from models import models # This imports the models module
from models.models import User, EnhancementHistoryBase, ImageCreate # Added User, EnhancementHistoryBase, ImageCreate
import logging # Added logging
from typing import List # Added for FaceDetection models

from services.face_detection import detect_faces # Added for face detection endpoint
from services.image_quality import analyze_image_quality # New import
from services.auto_enhancement import calculate_auto_enhancements # New import for auto enhancement
from services.image_processing import apply_enhancements, ImageProcessingError # New import for applying enhancements

from routers import auth as auth_router # Import the auth router
from routers import users as users_router # Import the users router
from auth_utils import get_current_user # Added for current_user dependency

# Pydantic model for content moderation result
class ContentModerationResult(BaseModel): # Corrected to BaseModel
    is_approved: bool
    rejection_reason: Optional[str] = None

# Pydantic models for Face Detection API
class FaceBoundingBox(BaseModel):
    box: List[int]  # [x, y, width, height]
    confidence: Optional[float] = None

class FaceDetectionResponse(BaseModel):
    faces: List[FaceBoundingBox]
    image_id: uuid.UUID # Changed from int
    message: Optional[str] = None

class ImageQualityMetrics(BaseModel):
    brightness: float
    contrast: float

class ImageQualityAnalysisResponse(BaseModel):
    image_id: uuid.UUID
    quality_metrics: ImageQualityMetrics
    insights: List[str]
    message: Optional[str] = None

# Pydantic models for Image Enhancement
class EnhancementParameters(BaseModel):
    brightness_target: float
    contrast_target: float
    saturation_target: float
    background_blur_radius: int
    crop_rect: List[int]  # [x, y, width, height]
    face_smooth_intensity: float

class AutoEnhancementResponse(BaseModel):
    image_id: uuid.UUID
    enhancement_parameters: EnhancementParameters
    message: Optional[str] = None

# Pydantic models for Manual Image Enhancement Request
class EnhancementRequestParams(BaseModel): # Mirrors EnhancementParameters
    brightness_target: float
    contrast_target: float
    saturation_target: float
    background_blur_radius: int
    crop_rect: List[int]  # [x, y, width, height]
    face_smooth_intensity: float

class ImageEnhancementRequest(BaseModel):
    image_id: uuid.UUID
    parameters: EnhancementRequestParams

# Pydantic model for Apply Preset Request
class ApplyPresetRequest(BaseModel):
    image_id: uuid.UUID

# Pydantic model for Processed Image Response
class ProcessedImageResponse(BaseModel):
    original_image_id: uuid.UUID
    processed_image_id: Optional[uuid.UUID] = None
    processed_image_path: Optional[str] = None
    message: str
    error: Optional[str] = None


# Maximum request body size (e.g., 1MB for general JSON, file uploads are separate)
MAX_REQUEST_BODY_SIZE = 1 * 1024 * 1024  # 1MB

# Middleware for adding security headers
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseCall) -> Response:
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        # A starting point for CSP. Might need adjustment for specific frontend needs (CDNs, inline scripts, etc.)
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; object-src 'none'; frame-ancestors 'none';"
        # HSTS: Only enable if ALWAYS served over HTTPS.
        # Note: While this application sets the HSTS header, actual HTTPS enforcement (SSL termination,
        # HTTP to HTTPS redirection) should be handled by a reverse proxy (e.g., Nginx, Traefik)
        # or a cloud load balancer in a production environment.
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        return response

class RequestBodySizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, max_size: int):
        super().__init__(app)
        self.max_size = max_size

    async def dispatch(self, request: Request, call_next: RequestResponseCall) -> StarletteResponse:
        # Skip size check for specific paths like file uploads if they handle streaming separately
        # For example, if '/api/images/upload' handles large files directly.
        # This example applies the limit to most other JSON-based endpoints.
        if request.url.path == "/api/images/upload": # Assuming this is your upload endpoint
             return await call_next(request)

        # Check Content-Length header first
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > self.max_size:
                    logger.warning(f"Request body size ({content_length} bytes) exceeds limit of {self.max_size} bytes for {request.url.path}.")
                    return StarletteResponse(content="Request entity too large.", status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, media_type="text/plain")
            except ValueError:
                logger.warning(f"Invalid Content-Length header: {content_length}")
                # Proceed to stream reading if content-length is invalid

        body_chunks = []
        received_size = 0
        stream = request.stream()
        async for chunk in stream:
            received_size += len(chunk)
            if received_size > self.max_size:
                logger.warning(f"Request body stream exceeded limit of {self.max_size} bytes at {request.url.path}.")
                return StarletteResponse(content="Request entity too large.", status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, media_type="text/plain")
            body_chunks.append(chunk)

        full_body = b"".join(body_chunks)

        # Replace the request's receive channel with one that returns the buffered body
        async def new_receive():
            return {"type": "http.request", "body": full_body, "more_body": False}

        request.scope['receive'] = new_receive

        response = await call_next(request)
        return response

# Optional Bearer token scheme for rate limiter key function
oauth2_scheme_optional = HTTPBearer(auto_error=False)

# Define rate limit strings
AUTH_USER_RATE_LIMIT = "100/minute"
ANON_USER_RATE_LIMIT = "20/minute"

# New key function for rate limiting (modified to add prefixes)
# This is the corrected and single version of this function.
def get_request_identifier_for_rate_limit(request: Request, creds: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_scheme_optional)) -> str:
    identifier_type = "ip" # To help with logging prefix
    identifier = request.client.host
    try:
        if creds and creds.credentials:
            token = creds.credentials
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id = payload.get("user_id")
            if user_id:
                identifier = str(user_id)
                identifier_type = "user"
                # logger.debug(f"Rate limiting by user_id: {identifier}") # Covered by prefix log
            else:
                sub = payload.get("sub")
                if sub:
                    identifier = str(sub)
                    identifier_type = "sub"
                    # logger.debug(f"Rate limiting by token subject (sub): {identifier}") # Covered by prefix log
                else:
                    logger.warning("Token present but no 'user_id' or 'sub' claim found. Falling back to IP.")
            return f"{identifier_type}:{identifier}"
    except JWTError as e:
        logger.warning(f"JWTError during token decoding for rate limit key: {e}. Falling back to IP.")
    except Exception as e:
        logger.error(f"Unexpected error in get_request_identifier_for_rate_limit: {e}. Falling back to IP.")

    # Fallback to IP if token processing failed or no token
    key = f"ip:{request.client.host}"
    request.state.rate_limit_key = key # Store for header middleware
    logger.debug(f"Rate limiting by {key} for request to {request.url.path}")
    return key


# Function to get dynamic rate limit based on key prefix
def get_dynamic_rate_limit(key: str) -> str:
    if key.startswith("user:") or key.startswith("sub:"):
        logger.debug(f"Applying authenticated user rate limit for key: {key}")
        return AUTH_USER_RATE_LIMIT
    else: # Starts with "ip:" or is an unexpected format (default to anon)
        if not key.startswith("ip:"):
            logger.warning(f"Unexpected key format '{key}' for dynamic rate limit, applying anonymous limit.")
        logger.debug(f"Applying anonymous user rate limit for key: {key}")
        return ANON_USER_RATE_LIMIT

# Initialize Limiter with the new key function
limiter = Limiter(key_func=get_request_identifier_for_rate_limit)

# Middleware for adding X-RateLimit headers
class RateLimitHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseCall) -> StarletteResponse:
        response = await call_next(request)

        key = getattr(request.state, 'rate_limit_key', None)

        # Only add headers if the key was set (meaning a rate-limited route was hit)
        # And if the app.state.limiter is available (it should be)
        if key and hasattr(request.app.state, 'limiter'):
            try:
                limiter_instance: Limiter = request.app.state.limiter

                # Determine the rate limit string dynamically for this key
                # This assumes that get_dynamic_rate_limit is accessible here.
                # It's defined globally in main.py, so it should be.
                rate_string = get_dynamic_rate_limit(key)

                # Parse the rate string to a Limit object (actually a list of them)
                # We'll use the first one, assuming simple limits like "100/minute"
                current_limit_obj_list = parse_many(rate_string)
                if not current_limit_obj_list:
                    logger.error(f"Could not parse rate string: {rate_string} for key {key}")
                    return response

                current_limit_obj: Limit = current_limit_obj_list[0]

                # Get window stats
                # get_window_stats expects a slowapi.models.Limit object, not just the string.
                window_stats = limiter_instance.storage.get_window_stats(key, current_limit_obj)

                reset_time = int(window_stats[0])
                remaining_count = window_stats[1]
                limit_amount = current_limit_obj.amount

                response.headers["X-RateLimit-Limit"] = str(limit_amount)
                response.headers["X-RateLimit-Remaining"] = str(remaining_count)
                response.headers["X-RateLimit-Reset"] = str(reset_time)
                logger.debug(f"Added rate limit headers for key {key}: Limit={limit_amount}, Remaining={remaining_count}, Reset={reset_time}")

            except Exception as e:
                logger.error(f"Error adding rate limit headers: {e}", exc_info=True)

        return response

# Custom RateLimitExceeded handler to add headers
async def custom_rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """
    Custom handler for RateLimitExceeded to add X-RateLimit headers.
    """
    response = JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": f"Rate limit exceeded: {exc.detail}"}
    )

    try:
        key = exc.key if hasattr(exc, 'key') else getattr(request.state, 'rate_limit_key', None)
        limit_obj = exc.limit if hasattr(exc, 'limit') else None

        if key and limit_obj and hasattr(request.app.state, 'limiter'):
            limiter_instance: Limiter = request.app.state.limiter
            window_stats = limiter_instance.storage.get_window_stats(key, limit_obj) # exc.limit should be a Limit object

            reset_time = int(window_stats[0])
            # Remaining is 0 because the limit was exceeded

            response.headers["X-RateLimit-Limit"] = str(limit_obj.amount)
            response.headers["X-RateLimit-Remaining"] = "0"
            response.headers["X-RateLimit-Reset"] = str(reset_time)
            logger.debug(f"Added rate limit headers to 429 response for key {key}")
        else:
            logger.warning(f"Could not add rate limit headers to 429: key or limit_obj missing. Key from state: {key}, exc.limit: {limit_obj}")

    except Exception as e:
        logger.error(f"Error in custom_rate_limit_exceeded_handler adding headers: {e}", exc_info=True)

    return response


app = FastAPI()

# Add Rate Limiter state and exception handler (custom one now)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, custom_rate_limit_exceeded_handler)


# Add middlewares to the application
# CORS should typically be one of the first, if not the first.
# However, if other middlewares might modify requests in ways that CORS should see,
# or if CORS needs to act before them, adjust order.
# For typical setup, CORS first is common.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Placeholder: Should be restricted in production via env var
    allow_credentials=True,
    allow_methods=["*"], # Allows all standard methods
    allow_headers=["*"], # Allows all headers
)

# SecurityHeadersMiddleware should ideally be one of the first
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestBodySizeLimitMiddleware, max_size=MAX_REQUEST_BODY_SIZE)
# RateLimitHeaderMiddleware should be added after slowapi might have interacted (though it doesn't use middleware)
# and after request.state.rate_limit_key is potentially set.
# It should be one of the later middlewares so it can modify the final response.
app.add_middleware(RateLimitHeaderMiddleware)


# It's important that routers are included AFTER the limiter is set on app.state
# and exception handlers are added.
app.include_router(auth_router.router) # Include the auth router
app.include_router(users_router.router) # Include the users router

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
UPLOAD_DIR = "uploads/images/"
PROCESSED_UPLOAD_DIR = "uploads/processed/" # Directory for processed images
ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"]
MIME_TYPE_TO_EXTENSION = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
MAX_FILE_SIZE_MB = 15 # Max for image files (handled by endpoint)
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
# MAX_REQUEST_BODY_SIZE is defined above for general requests

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
@limiter.limit(get_dynamic_rate_limit)
async def upload_image(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
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

    # Sanitize the filename before using it
    sanitized_filename = secure_filename(file.filename)
    if not sanitized_filename: # Handle cases where filename might be empty or only contain invalid chars
        sanitized_filename = f"upload_{uuid.uuid4().hex}" # Fallback filename

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
        filename=sanitized_filename, # Use sanitized filename
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
        logger.info(f"Image rejected: {moderation_result.rejection_reason}. Original filename: {file.filename}, Sanitized: {sanitized_filename}")
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

# Endpoint for Face Detection
@app.get("/api/analysis/faces/{image_id}", response_model=FaceDetectionResponse)
@limiter.limit(get_dynamic_rate_limit)
async def get_face_detections(request: Request, image_id: uuid.UUID, db: Session = Depends(get_db)): # Changed image_id type to uuid.UUID
    db_image = crud.get_image(db, image_id=image_id) # crud.get_image should handle UUID
    if not db_image:
        logger.warning(f"Face detection request for non-existent image_id: {image_id}")
        raise HTTPException(status_code=404, detail=f"Image with id {image_id} not found.")

    if not db_image.filepath:
        # This case might occur if an image was recorded in DB but file saving failed
        # or if it's a rejected image that wasn't saved (e.g. content moderation failed)
        logger.info(f"Filepath for image id {image_id} not available. Image status: {'approved' if not db_image.rejection_reason else f'rejected ({db_image.rejection_reason})'}")
        raise HTTPException(status_code=404, detail=f"Filepath for image id {image_id} not available. The image might not have been processed or saved correctly.")

    try:
        logger.info(f"Attempting face detection for image id {image_id} at path: {db_image.filepath}")
        detected_faces_data = detect_faces(db_image.filepath)

        response_faces = []
        for face_data in detected_faces_data:
            response_faces.append(FaceBoundingBox(box=face_data['box'], confidence=face_data.get('confidence')))

        if not response_faces:
            logger.info(f"No faces detected for image id {image_id} using local detection.")
            return FaceDetectionResponse(faces=[], image_id=db_image.id, message="No faces detected.") # Use db_image.id (UUID)

        logger.info(f"Successfully detected {len(response_faces)} faces for image id {image_id} using local detection.")
        return FaceDetectionResponse(faces=response_faces, image_id=db_image.id) # Use db_image.id (UUID)

    except FileNotFoundError:
        logger.error(f"File not found for image id {image_id} at path: {db_image.filepath} during face detection attempt.")
        raise HTTPException(status_code=404, detail=f"Image file not found for id {image_id}. It may have been moved or deleted after upload.")
    except ValueError as e:
        # This can be caught if detect_faces raises it due to image loading issues (e.g. invalid image file)
        logger.error(f"Error processing image id {image_id} with local face detection (ValueError): {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image id {image_id}: Invalid image file or format. Details: {str(e)}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while detecting faces for image id {image_id} using local detection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during face detection.")


@app.get("/api/analysis/quality/{image_id}", response_model=ImageQualityAnalysisResponse)
@limiter.limit(get_dynamic_rate_limit)
async def get_image_quality_analysis(request: Request, image_id: uuid.UUID, db: Session = Depends(get_db)):
    db_image = crud.get_image(db, image_id=image_id)
    if not db_image:
        logger.warning(f"Image quality analysis request for non-existent image_id: {image_id}")
        raise HTTPException(status_code=404, detail=f"Image with id {image_id} not found.")

    if not db_image.filepath:
        logger.info(f"Filepath for image id {image_id} not available for quality analysis. Image status: {'approved' if not db_image.rejection_reason else f'rejected ({db_image.rejection_reason})'}")
        raise HTTPException(status_code=404, detail=f"Filepath for image id {image_id} not available. The image might not have been processed or saved correctly.")

    if not os.path.exists(db_image.filepath): # Check if file exists before analysis
        logger.error(f"File not found for image id {image_id} at path: {db_image.filepath} during quality analysis attempt.")
        raise HTTPException(status_code=404, detail=f"Image file not found at path {db_image.filepath}. It may have been moved or deleted.")

    try:
        logger.info(f"Attempting image quality analysis for image id {image_id} at path: {db_image.filepath}")

        # Call the analysis function from the service
        quality_data = analyze_image_quality(db_image.filepath)

        metrics = ImageQualityMetrics(
            brightness=quality_data["brightness"],
            contrast=quality_data["contrast"]
        )

        return ImageQualityAnalysisResponse(
            image_id=db_image.id,
            quality_metrics=metrics,
            insights=quality_data["insights"],
            message="Image quality analysis successful."
        )

    except FileNotFoundError: # Should be caught by the os.path.exists check, but good to have
        logger.error(f"File not found for image id {image_id} at path: {db_image.filepath} during quality analysis.")
        raise HTTPException(status_code=404, detail=f"Image file not found for id {image_id} during analysis.")
    except ValueError as e:
        logger.error(f"Error processing image id {image_id} with quality analysis (ValueError): {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image id {image_id}: Invalid image file or format. Details: {str(e)}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while analyzing image quality for id {image_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during image quality analysis.")


@app.get("/api/enhancement/auto/{image_id}", response_model=AutoEnhancementResponse)
@limiter.limit(get_dynamic_rate_limit)
async def get_auto_enhancement_parameters(
    request: Request,
    image_id: uuid.UUID,
    mode: Optional[str] = None, # Allows for different enhancement modes e.g. "passport"
    db: Session = Depends(get_db)
):
    logger.info(f"Received request for auto enhancement parameters for image_id: {image_id}, mode: {mode}")

    db_image = crud.get_image(db, image_id=image_id)
    if not db_image:
        logger.warning(f"Image not found in DB for id: {image_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Image with id {image_id} not found.")

    if not db_image.filepath:
        logger.warning(f"Filepath not available for image_id: {image_id}. Image status: {'approved' if not db_image.rejection_reason else f'rejected ({db_image.rejection_reason})'}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Filepath for image id {image_id} not available. Image may not have been processed or saved.")

    if not os.path.exists(db_image.filepath):
        logger.error(f"Image file not found at path: {db_image.filepath} for image_id: {image_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Image file not found at path {db_image.filepath}. It may have been moved or deleted.")

    if db_image.width is None or db_image.height is None:
        logger.error(f"Image dimensions are missing for image_id: {image_id}. Width: {db_image.width}, Height: {db_image.height}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Image dimensions (width, height) missing for image id {image_id}. Cannot perform enhancements.")
    image_dimensions = (db_image.width, db_image.height)

    face_results = []
    try:
        logger.info(f"Performing face detection for image_id: {image_id} at path: {db_image.filepath}")
        face_results = detect_faces(db_image.filepath) # Returns a list of dicts
    except FileNotFoundError: # Should be caught by os.path.exists, but good as a safeguard
        logger.error(f"Face detection: File not found for image_id: {image_id} at path: {db_image.filepath}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Image file not found during face detection for id {image_id}.")
    except ValueError as e: # If detect_faces raises ValueError for bad image
        logger.error(f"Face detection: Error processing image_id {image_id} (ValueError): {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing image for face detection (id: {image_id}): {str(e)}")
    except Exception as e:
        logger.error(f"Face detection: Unexpected error for image_id {image_id}: {e}", exc_info=True)
        # Depending on policy, we might allow proceeding without face_results or raise 500.
        # For now, let's proceed, calculate_auto_enhancements should handle empty face_results.
        # If it's critical, raise HTTPException here. For now, it defaults to empty list.
        pass # Proceed with empty face_results

    quality_results = {}
    try:
        logger.info(f"Performing image quality analysis for image_id: {image_id} at path: {db_image.filepath}")
        quality_results = analyze_image_quality(db_image.filepath) # Returns a dict
    except FileNotFoundError: # Should be caught by os.path.exists
        logger.error(f"Quality analysis: File not found for image_id: {image_id} at path: {db_image.filepath}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Image file not found during quality analysis for id {image_id}.")
    except ValueError as e: # If analyze_image_quality raises ValueError for bad image
        logger.error(f"Quality analysis: Error processing image_id {image_id} (ValueError): {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing image for quality analysis (id: {image_id}): {str(e)}")
    except Exception as e:
        logger.error(f"Quality analysis: Unexpected error for image_id {image_id}: {e}", exc_info=True)
        # If quality_results are essential and cannot be defaulted in calculate_auto_enhancements, raise 500.
        # calculate_auto_enhancements has defaults if keys are missing.
        # So, we can proceed with potentially empty quality_results or partial data.
        pass # Proceed with potentially empty/partial quality_results


    logger.info(f"Calculating auto enhancement parameters for image_id: {image_id} with mode: {mode}")
    try:
        enhancement_params_dict = calculate_auto_enhancements(
            image_path=db_image.filepath,
            image_dimensions=image_dimensions,
            face_detection_results=face_results,
            image_quality_results=quality_results,
            mode=mode
        )
        enhancement_params_model = EnhancementParameters(**enhancement_params_dict)

        logger.info(f"Successfully calculated enhancement parameters for image_id: {image_id}")
        return AutoEnhancementResponse(
            image_id=image_id,
            enhancement_parameters=enhancement_params_model,
            message="Auto enhancement parameters calculated successfully."
        )
    except Exception as e:
        logger.error(f"Error calculating auto enhancement parameters for image_id {image_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to calculate auto enhancement parameters due to an internal error.")

@app.post("/api/enhancement/apply", response_model=ProcessedImageResponse)
@limiter.limit(get_dynamic_rate_limit)
async def apply_image_enhancements_endpoint(
    # The 'request: Request' parameter for FastAPI/Starlette must come before `request: ImageEnhancementRequest` (Pydantic model)
    # So, we rename one of them. Let's rename the FastAPI/Starlette request.
    http_request: Request, # Renamed to avoid conflict with Pydantic model 'request'
    request: ImageEnhancementRequest, # This is the Pydantic model for the request body
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user) # Added current_user
):
    logger.info(f"Received request to apply enhancements for image_id: {request.image_id} by user: {current_user.id}") # Log user

    db_image = crud.get_image(db, image_id=request.image_id)
    if not db_image:
        logger.warning(f"Apply Enhancement: Image not found in DB for id: {request.image_id}")
        # Return 200 OK with error message in body as per ProcessedImageResponse model for client handling
        return ProcessedImageResponse(
            original_image_id=request.image_id,
            message="Failed to apply enhancements.",
            error=f"Image with id {request.image_id} not found."
        )

    if not db_image.filepath:
        logger.warning(f"Apply Enhancement: Filepath not available for image_id: {request.image_id}")
        return ProcessedImageResponse(
            original_image_id=request.image_id,
            message="Failed to apply enhancements.",
            error=f"Filepath for image id {request.image_id} not available."
        )

    if not os.path.exists(db_image.filepath):
        logger.error(f"Apply Enhancement: Image file not found at path: {db_image.filepath} for image_id: {request.image_id}")
        return ProcessedImageResponse(
            original_image_id=request.image_id,
            message="Failed to apply enhancements.",
            error=f"Image file not found at path {db_image.filepath}."
        )

    face_results = []
    # Perform face detection if face smoothing or other face-dependent features are requested
    # For simplicity, we run it if smoothing intensity is positive.
    # Could also run if auto-crop based on faces is a parameter (not currently the case for manual apply).
    if request.parameters.face_smooth_intensity > 0: # or other face-dependent params
        try:
            logger.info(f"Apply Enhancement: Performing face detection for image_id: {request.image_id} for smoothing.")
            face_results = detect_faces(db_image.filepath)
        except FileNotFoundError:
             logger.error(f"Apply Enhancement: File not found for face detection: {db_image.filepath}")
             return ProcessedImageResponse(
                original_image_id=request.image_id,
                message="Failed to apply enhancements.",
                error="Original image file not found during face detection."
            )
        except Exception as e:
            logger.error(f"Apply Enhancement: Error during face detection for image_id {request.image_id}: {e}", exc_info=True)
            # Decide if this is fatal. For now, proceed with no faces.
            face_results = []


    processed_pil_image: Optional[PILImage.Image] = None
    try:
        logger.info(f"Apply Enhancement: Calling apply_enhancements service for image_id: {request.image_id}")
        processed_pil_image = apply_enhancements(
            image_path=db_image.filepath,
            params=request.parameters.model_dump(), # Convert Pydantic model to dict
            face_detection_results=face_results
        )
    except ImageProcessingError as e:
        logger.error(f"Apply Enhancement: ImageProcessingError for image_id {request.image_id}: {e}", exc_info=True)
        return ProcessedImageResponse(
            original_image_id=request.image_id,
            message="Failed to apply enhancements due to processing error.",
            error=str(e)
        )
    except Exception as e:
        logger.error(f"Apply Enhancement: Unexpected error during image processing for image_id {request.image_id}: {e}", exc_info=True)
        return ProcessedImageResponse(
            original_image_id=request.image_id,
            message="Failed to apply enhancements due to an unexpected server error.",
            error=str(e)
        )

    if processed_pil_image is None:
        # This case should ideally be caught by specific exceptions above, but as a safeguard:
        logger.error(f"Apply Enhancement: Processing returned None for image_id: {request.image_id}")
        return ProcessedImageResponse(
            original_image_id=request.image_id,
            message="Failed to apply enhancements.",
            error="Image processing returned no result."
        )

    try:
        os.makedirs(PROCESSED_UPLOAD_DIR, exist_ok=True)
        # Save as PNG to preserve quality and handle alpha channels from blur/segmentation
        new_filename = f"{db_image.id}_enhanced_{uuid.uuid4().hex}.png"
        processed_image_filepath = os.path.join(PROCESSED_UPLOAD_DIR, new_filename)

        logger.info(f"Apply Enhancement: Saving processed image for id {request.image_id} to {processed_image_filepath}")
        processed_pil_image.save(processed_image_filepath, format='PNG')

        # Initialize variables for processed image record and ID
        db_processed_image_record = None
        processed_image_id_for_response = None

        try:
            # 3. Create an Image record for the processed image
            processed_image_filesize = os.path.getsize(processed_image_filepath)
            processed_image_width, processed_image_height = processed_pil_image.size
            processed_image_format = processed_pil_image.format if processed_pil_image.format else 'PNG'


            image_create_data = ImageCreate(
                filename=new_filename,
                filepath=processed_image_filepath,
                filesize=processed_image_filesize,
                mimetype='image/png', # Explicitly PNG as we save in this format
                width=processed_image_width,
                height=processed_image_height,
                format=processed_image_format
                # exif_orientation and color_profile can be None or copied if available/relevant
            )
            db_processed_image_record = crud.create_image(
                db=db,
                image=image_create_data,
                width=image_create_data.width,
                height=image_create_data.height,
                format=image_create_data.format
            )
            processed_image_id_for_response = db_processed_image_record.id
            logger.info(f"Apply Enhancement: Created DB record for processed image with ID: {db_processed_image_record.id}")

            # 5. Create EnhancementHistory record
            parameters_json_str = json.dumps(request.parameters.model_dump())
            history_create_data = EnhancementHistoryBase(
                original_image_id=db_image.id, # original image ID
                processed_image_id=db_processed_image_record.id, # new processed image ID
                parameters_json=parameters_json_str
            )
            crud.create_enhancement_history(db=db, history_data=history_create_data, user_id=current_user.id)
            logger.info(f"Apply Enhancement: Enhancement history record created for user {current_user.id}, original image {db_image.id}, processed image {db_processed_image_record.id}")

        except Exception as db_error:
            # Log DB errors but don't fail the entire request if file was saved
            logger.error(f"Apply Enhancement: Error during DB operations (saving processed image record or history) for original image id {request.image_id}: {db_error}", exc_info=True)
            # The processed_image_id_for_response might be None if create_image failed, or set if history creation failed.
            # This is acceptable as per error handling guidelines (prioritize returning image path).

        return ProcessedImageResponse(
            original_image_id=request.image_id,
            processed_image_id=processed_image_id_for_response, # This will be None if DB op failed before setting it
            processed_image_path=processed_image_filepath,
            message="Image enhancements applied and saved successfully."
        )
    except IOError as e:
        logger.error(f"Apply Enhancement: Failed to save processed image for id {request.image_id} to path {processed_image_filepath}: {e}", exc_info=True)
        return ProcessedImageResponse(
            original_image_id=request.image_id,
            message="Failed to save processed image after applying enhancements.",
            error=str(e)
        )
    except Exception as e: # Catch any other unexpected errors during save
        logger.error(f"Apply Enhancement: Unexpected error saving processed image for {request.image_id}: {e}", exc_info=True)
        return ProcessedImageResponse(
            original_image_id=request.image_id,
            message="An unexpected error occurred while saving the processed image.",
            error=str(e)
        )


@app.post("/api/enhancement/apply-preset/{preset_id}", response_model=ProcessedImageResponse)
@limiter.limit(get_dynamic_rate_limit)
async def apply_preset_to_image_endpoint(
    http_request: Request, # Renamed to avoid conflict
    preset_id: uuid.UUID,
    request_data: ApplyPresetRequest, # This is the Pydantic model for the request body
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    logger.info(f"User {current_user.id} applying preset {preset_id} to image {request_data.image_id}")

    # 1. Fetch the User Preset
    db_preset = crud.get_user_preset(db, preset_id=preset_id, user_id=current_user.id)
    if not db_preset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preset not found or not owned by user")

    # 2. Parse Preset Parameters
    try:
        preset_parameters_dict = json.loads(db_preset.parameters_json)
        # Validate/convert to EnhancementRequestParams Pydantic model
        enhancement_params = EnhancementRequestParams(**preset_parameters_dict)
    except json.JSONDecodeError:
        logger.error(f"Preset {preset_id} has invalid JSON parameters: {db_preset.parameters_json}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Preset contains invalid parameters format.")
    except Exception as e: # Catches Pydantic validation errors too
        logger.error(f"Preset {preset_id} parameters validation failed: {e}. Parameters: {db_preset.parameters_json}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Preset parameters are invalid: {e}")

    # 3. Fetch the Original Image (similar to apply_image_enhancements_endpoint)
    db_image = crud.get_image(db, image_id=request_data.image_id)
    if not db_image:
        return ProcessedImageResponse(original_image_id=request_data.image_id, message="Failed to apply preset.", error=f"Image with id {request_data.image_id} not found.")
    if not db_image.filepath:
        return ProcessedImageResponse(original_image_id=request_data.image_id, message="Failed to apply preset.", error=f"Filepath for image id {request_data.image_id} not available.")
    if not os.path.exists(db_image.filepath):
        return ProcessedImageResponse(original_image_id=request_data.image_id, message="Failed to apply preset.", error=f"Image file not found at path {db_image.filepath}.")

    # 4. Face Detection (if needed)
    face_results = []
    if enhancement_params.face_smooth_intensity > 0:
        try:
            face_results = detect_faces(db_image.filepath)
        except Exception as e:
            logger.error(f"Apply Preset: Error during face detection for image {request_data.image_id}: {e}", exc_info=True)
            # Proceed without face_results, or return error if critical
            face_results = []

    # 5. Apply Enhancements
    processed_pil_image: Optional[PILImage.Image] = None
    try:
        processed_pil_image = apply_enhancements(
            image_path=db_image.filepath,
            params=enhancement_params.model_dump(),
            face_detection_results=face_results
        )
    except ImageProcessingError as e:
        logger.error(f"Apply Preset: ImageProcessingError for image {request_data.image_id} with preset {preset_id}: {e}", exc_info=True)
        return ProcessedImageResponse(original_image_id=request_data.image_id, message="Failed to apply preset due to processing error.", error=str(e))
    except Exception as e:
        logger.error(f"Apply Preset: Unexpected error during image processing for image {request_data.image_id} with preset {preset_id}: {e}", exc_info=True)
        return ProcessedImageResponse(original_image_id=request_data.image_id, message="Failed to apply preset due to an unexpected server error.", error=str(e))

    if processed_pil_image is None:
        return ProcessedImageResponse(original_image_id=request_data.image_id, message="Failed to apply preset.", error="Image processing returned no result.")

    # 6. Save Processed Image and Record History
    try:
        os.makedirs(PROCESSED_UPLOAD_DIR, exist_ok=True)
        new_filename = f"{db_image.id}_preset_enhanced_{uuid.uuid4().hex}.png"
        processed_image_filepath = os.path.join(PROCESSED_UPLOAD_DIR, new_filename)
        processed_pil_image.save(processed_image_filepath, format='PNG')

        db_processed_image_record = None
        processed_image_id_for_response = None
        try:
            processed_image_filesize = os.path.getsize(processed_image_filepath)
            processed_image_width, processed_image_height = processed_pil_image.size
            img_create_data = ImageCreate(
                filename=new_filename, filepath=processed_image_filepath, filesize=processed_image_filesize,
                mimetype='image/png', width=processed_image_width, height=processed_image_height,
                format=processed_pil_image.format if processed_pil_image.format else 'PNG'
            )
            db_processed_image_record = crud.create_image(db=db, image=img_create_data, width=img_create_data.width, height=img_create_data.height, format=img_create_data.format)
            processed_image_id_for_response = db_processed_image_record.id
            logger.info(f"Apply Preset: Created DB record for processed image {db_processed_image_record.id}")

            # Create EnhancementHistory record, noting the preset used
            history_params_dict = enhancement_params.model_dump()
            history_params_dict["applied_preset_id"] = str(preset_id) # Add preset ID to history
            parameters_json_for_history = json.dumps(history_params_dict)

            history_create_data = EnhancementHistoryBase(
                original_image_id=db_image.id,
                processed_image_id=db_processed_image_record.id,
                parameters_json=parameters_json_for_history
            )
            crud.create_enhancement_history(db=db, history_data=history_create_data, user_id=current_user.id)
            logger.info(f"Apply Preset: Enhancement history created for user {current_user.id}, original image {db_image.id}, processed {db_processed_image_record.id}, preset {preset_id}")

        except Exception as db_error:
            logger.error(f"Apply Preset: DB error for original image {request_data.image_id}, preset {preset_id}: {db_error}", exc_info=True)
            # Return image path even if DB history fails

        return ProcessedImageResponse(
            original_image_id=request_data.image_id,
            processed_image_id=processed_image_id_for_response,
            processed_image_path=processed_image_filepath,
            message="Preset applied and image saved successfully."
        )
    except IOError as e:
        logger.error(f"Apply Preset: Failed to save processed image for {request_data.image_id}, preset {preset_id}: {e}", exc_info=True)
        return ProcessedImageResponse(original_image_id=request_data.image_id, message="Failed to save processed image after applying preset.", error=str(e))
    except Exception as e:
        logger.error(f"Apply Preset: Unexpected error saving image for {request_data.image_id}, preset {preset_id}: {e}", exc_info=True)
        return ProcessedImageResponse(original_image_id=request_data.image_id, message="An unexpected error occurred while saving the processed image using preset.", error=str(e))
