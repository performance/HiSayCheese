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
from starlette.middleware.base import BaseHTTPMiddleware # RequestResponseCall removed
from starlette.requests import Request
from starlette.responses import Response as StarletteResponse # Renamed to avoid conflict with FastAPI's Response
from starlette.types import ASGIApp # Added for middleware
from typing import Callable, Awaitable, Any # Added for RequestResponseCall definition and Any type
import time # For ResponseTimeLoggingMiddleware

# Define RequestResponseCall if not available from starlette directly
RequestResponseCall = Callable[[Request], Awaitable[StarletteResponse]]

# Rate Limiting - Moved to rate_limiter.py
# from slowapi import Limiter, _rate_limit_exceeded_handler # Now in rate_limiter.py
from slowapi.errors import RateLimitExceeded # Still needed for exception handler
# from limits.util import parse_many # Now in rate_limiter.py
from rate_limiter import limiter, get_dynamic_rate_limit, ANON_USER_RATE_LIMIT, AUTH_USER_RATE_LIMIT # Import from new module

from fastapi.responses import JSONResponse # For the exception handler
# HTTPBearer, HTTPAuthorizationCredentials, jwt, JWTError, SECRET_KEY, ALGORITHM are used by rate_limiter.py
# No longer need to be directly imported in main.py IF they are only for rate limiting logic.
# However, get_request_identifier_for_rate_limit in main.py (if it were still here) would need them.
# Let's check if they are used elsewhere in main.py...
# SECRET_KEY, ALGORITHM are used by RequestIdMiddleware for user_id logging.
# HTTPBearer, HTTPAuthorizationCredentials, jwt, JWTError are also used by RequestIdMiddleware.
# So, these specific imports need to remain in main.py for RequestIdMiddleware.
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from config import SECRET_KEY, ALGORITHM


from db.database import create_db_and_tables, get_db
from db import crud
from models import models # This imports the models module
# We need ImageCreate and ImageSchema for type hinting and response_model
# from models.models import ImageCreate, ImageSchema # More specific imports
# Corrected import for models to include User, EnhancementHistoryBase, ImageCreate
from models import models # This imports the models module
from models.models import User, EnhancementHistoryBase, ImageCreate # Added User, EnhancementHistoryBase, ImageCreate
# import logging # Added logging
from pythonjsonlogger import jsonlogger # For JSON logging
import logging # Standard logging
import sentry_sdk # For Sentry integration
from sentry_sdk.integrations.logging import LoggingIntegration # Sentry logging integration
from typing import List # Added for FaceDetection models

# Import SENTRY_DSN from config
from config import SENTRY_DSN

from services.face_detection import detect_faces # Added for face detection endpoint
from services.image_quality import analyze_image_quality # New import
from services.auto_enhancement import calculate_auto_enhancements # New import for auto enhancement
from services.image_processing import apply_enhancements, ImageProcessingError # New import for applying enhancements
from services.storage_service import StorageService # Import StorageService
from io import BytesIO # Import BytesIO

from routers import auth as auth_router # Import the auth router
from routers import users as users_router # Import the users router
from routers import health as health_router # Import the health router
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

# Middleware for logging response times
class ResponseTimeLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseCall) -> StarletteResponse:
        start_time = time.time()
        response = await call_next(request)
        process_time_ms = (time.time() - start_time) * 1000

        # logger instance is already defined globally in main.py (logger = logging.getLogger(__name__))
        # RequestIdMiddleware already adds request_id and user_id to log records if available.
        # So, we just need to log the additional details.

        log_details = {
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "response_time_ms": round(process_time_ms, 2)
        }
        # request_id and user_id will be added by RequestIdMiddleware's factory if this log goes through it.
        # If this middleware is *after* RequestIdMiddleware, these attributes are already on request.state
        # and thus will be picked up by the custom log record factory.
        # To be more robust, explicitly pass request_id and user_id if available.
        log_details["request_id"] = getattr(request.state, 'request_id', None)
        log_details["user_id"] = getattr(request.state, 'user_id', None)
        logger.info("Request processed", extra=log_details)

        return response

# Middleware for adding Request ID
class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseCall) -> StarletteResponse:
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id

        # Attempt to get user_id from token
        user_id_for_log = None
        try:
            token_creds = await HTTPBearer(auto_error=False)(request)
            if token_creds and token_creds.credentials:
                token = token_creds.credentials
                payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                user_id_for_log = payload.get("user_id") or payload.get("sub")
        except JWTError:
            # Token might be invalid or expired, or not present. Fine for logging.
            pass
        except Exception:
            # Other unexpected errors during token processing for logging.
            pass
        request.state.user_id = user_id_for_log

        original_log_record_factory = logging.getLogRecordFactory()

        # Keep track of whether we've set our custom factory for this request
        # to avoid issues if middleware is somehow re-entered or for nested calls.
        # This is a simple flag; more robust solutions might use contextvars for logger state.
        # For FastAPI/Starlette middleware, dispatch is usually called once per request.

        current_factory = logging.getLogRecordFactory()
        # Check if the factory is already our custom one (e.g. from a previous middleware instance, though unlikely for this setup)
        # This check is more illustrative; direct reset in `finally` is the key.
        is_custom_factory_active = hasattr(current_factory, '_is_request_id_factory')


        def new_log_record_factory(*args, **kwargs):
            record = original_log_record_factory(*args, **kwargs)
            # Ensure request.state attributes are accessed safely
            record.request_id = getattr(request.state, 'request_id', 'N/A')
            record.user_id = getattr(request.state, 'user_id', None)
            return record

        # Mark our factory so we could potentially identify it, though direct reset is better.
        # new_log_record_factory._is_request_id_factory = True

        logging.setLogRecordFactory(new_log_record_factory)

        try:
            response = await call_next(request)
        finally:
            # Always reset to the original factory captured at the start of this dispatch
            logging.setLogRecordFactory(original_log_record_factory)

        response.headers['X-Request-ID'] = request_id
        return response

# Middleware for adding security headers
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseCall) -> StarletteResponse: # Changed Response to StarletteResponse
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
        # Ensure all file upload endpoints are excluded.
        if request.url.path in ["/api/images/upload", "/api/users/upload-image"]:
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

# oauth2_scheme_optional, AUTH_USER_RATE_LIMIT, ANON_USER_RATE_LIMIT,
# get_request_identifier_for_rate_limit, get_dynamic_rate_limit, and limiter initialization
# are now in rate_limiter.py. We import 'limiter' and 'get_dynamic_rate_limit'.

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

                current_limit_obj: Any = current_limit_obj_list[0] # Type changed to Any

                # Get window stats
                # get_window_stats expects a parsed limit object from 'limits' library.
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
        # exc.limit is the specific limit object that was hit
        limit_obj_hit = exc.limit if hasattr(exc, 'limit') else None

        if key and limit_obj_hit and hasattr(request.app.state, 'limiter'):
            limiter_instance: Limiter = request.app.state.limiter
            # get_window_stats expects the specific limit object that was hit
            window_stats = limiter_instance.storage.get_window_stats(key, limit_obj_hit)

            reset_time = int(window_stats[0])
            # Remaining is 0 because the limit was exceeded

            response.headers["X-RateLimit-Limit"] = str(limit_obj_hit.amount)
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
# RequestIdMiddleware should be one of the very first, to ensure ID is available for all subsequent logs.
app.add_middleware(RequestIdMiddleware)
app.add_middleware(ResponseTimeLoggingMiddleware) # Added ResponseTimeLoggingMiddleware
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
app.include_router(health_router.router) # Include the health router

# Configure JSON logging
logger = logging.getLogger(__name__) # Get logger instance first
logger.setLevel(logging.INFO) # Set level for the logger instance

# Remove existing handlers if any (e.g., from basicConfig)
# This is important if basicConfig was called before or if running in an env that pre-configures logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
for handler in logger.handlers[:]: # Also clear handlers for the specific logger instance
    logger.removeHandler(handler)

log_handler = logging.StreamHandler()
# Updated formatter to include request_id, user_id, path, method, status_code, response_time_ms
formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(module)s %(funcName)s %(lineno)d %(message)s %(request_id)s %(user_id)s %(path)s %(method)s %(status_code)s %(response_time_ms)s')
log_handler.setFormatter(formatter)
logger.addHandler(log_handler)

# If you want the root logger to also use this format (e.g., for logs from other libraries)
# you might need to configure the root logger similarly.
# However, often it's better to configure only your application's logger.
# For now, let's also configure the root logger to ensure all logs are JSON.
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO) # Set level for root logger
# Remove any default handlers from root logger
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
root_logger.addHandler(log_handler) # Add our JSON handler to root

# logger = logging.getLogger(__name__) # This line is now redundant as logger is already configured

# Initialize Sentry
if SENTRY_DSN and SENTRY_DSN != "your-sentry-dsn-goes-here":
    sentry_logging = LoggingIntegration(
        level=logging.DEBUG,        # Breadcrumbs level
        event_level=logging.INFO    # Event level (INFO and above)
    )
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        integrations=[sentry_logging],
        traces_sample_rate=1.0, # Capture 100% of transactions for performance monitoring
        profiles_sample_rate=1.0 # Capture 100% of profiles for performance monitoring
    )
    logger.info("Sentry initialized.")
else:
    logger.warning("Sentry DSN not found or is placeholder. Sentry will not be initialized.")

# Instantiate StorageService
try:
    storage_service = StorageService()
    logger.info("StorageService initialized successfully.")
except HTTPException as e:
    logger.error(f"Failed to initialize StorageService: {e.detail}")
    # Depending on policy, we might re-raise or exit, or allow app to run with degraded functionality.
    # For now, let it proceed and fail on operations if storage_service is not available.
    # A more robust approach might involve a health check for S3 connectivity.
    storage_service = None # Ensure it's None if initialization fails
except Exception as e:
    logger.error(f"An unexpected error occurred during StorageService initialization: {e}")
    storage_service = None # Ensure it's None


# Constants
ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"]
MIME_TYPE_TO_EXTENSION = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
# Define a temporary directory for local processing if needed
TEMP_PROCESSING_DIR = "/tmp/image_processing/"
MAX_FILE_SIZE_MB = 15 # Max for image files (handled by endpoint)
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
# MAX_REQUEST_BODY_SIZE is defined above for general requests

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.get("/health", response_model=models.NumberSchema) # Corrected to NumberSchema
async def health(db: Session = Depends(get_db)):
    logger.info("Processing /health endpoint.") # ADD THIS LINE
    db_number = crud.get_number(db)
    if db_number is None:
        # logger.info("/health: No number set yet.") # Optional: good for debugging
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

@app.get("/test-error")
async def test_error_endpoint():
    try:
        raise ValueError("Test error for Sentry")
    except ValueError as e:
        logger.error("Intentional test error occurred", exc_info=True)
        # Re-raise or return error response, depending on desired test behavior
        # For testing Sentry, just logging it might be enough if Sentry is hooked into logging
        # For testing actual HTTP response, re-raise or return specific status
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Intentional test error")


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
        # Approved: Upload to S3
        if not storage_service:
            logger.error("StorageService not available for image upload.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Image storage service is not configured.")

        s3_object_key = f"original_images/{uuid.uuid4().hex}{file_extension}"

        try:
            # Use BytesIO to treat the 'contents' (bytes) as a file-like object for upload_fileobj
            storage_service.upload_file(
                file_obj=BytesIO(contents),
                object_key=s3_object_key,
                content_type=actual_content_type,
                acl="private" # Or your desired ACL
            )
            logger.info(f"Approved image uploaded to S3 with key: {s3_object_key}")
            image_data_to_create.filepath = s3_object_key # Update filepath to S3 object key
        except HTTPException as e: # Catch HTTPException from storage_service
            logger.error(f"S3 upload failed: {e.detail}")
            # Re-raise the HTTPException from storage_service
            raise e
        except Exception as e: # Catch any other unexpected errors during upload
            logger.error(f"An unexpected error occurred during S3 upload: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not upload file to S3: {e}",
            )

        # Ensure rejection_reason is explicitly None for approved images in the database
        image_data_to_create.rejection_reason = None

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

        presigned_url_for_response = None
        if storage_service and db_image.filepath: # db_image.filepath is the S3 key
            try:
                presigned_url_for_response = storage_service.generate_presigned_url(db_image.filepath)
                logger.info(f"Generated presigned URL for new upload {db_image.filepath}: {presigned_url_for_response}")
            except Exception as e_url:
                logger.error(f"Failed to generate presigned URL for new upload {db_image.filepath}: {e_url}", exc_info=True)
                # Non-critical error, proceed without presigned_url if generation fails

        # Construct response dictionary matching ImageSchema fields + new presigned_url field
        response_data = {
            "id": db_image.id,
            "filename": db_image.filename,
            "filepath": db_image.filepath, # This remains the S3 key
            "filesize": db_image.filesize,
            "mimetype": db_image.mimetype,
            "width": db_image.width,
            "height": db_image.height,
            "format": db_image.format,
            "exif_orientation": db_image.exif_orientation,
            "color_profile": db_image.color_profile,
            "rejection_reason": db_image.rejection_reason,
            "created_at": db_image.created_at,
            "presigned_url": presigned_url_for_response
        }
        return response_data # FastAPI will validate this against ImageSchema

# Endpoint for Face Detection
@app.get("/api/analysis/faces/{image_id}", response_model=FaceDetectionResponse)
@limiter.limit(get_dynamic_rate_limit)
async def get_face_detections(request: Request, image_id: uuid.UUID, db: Session = Depends(get_db)): # Changed image_id type to uuid.UUID
    db_image = crud.get_image(db, image_id=image_id) # crud.get_image should handle UUID
    if not db_image:
        logger.warning(f"Face detection request for non-existent image_id: {image_id}")
        raise HTTPException(status_code=404, detail=f"Image with id {image_id} not found.")

    if not db_image.filepath:
        logger.info(f"Filepath (S3 key) for image id {image_id} not available. Status: {'approved' if not db_image.rejection_reason else f'rejected ({db_image.rejection_reason})'}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"S3 key for image id {image_id} not available.")

    if not storage_service:
        logger.error("StorageService not available for face detection.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Image storage service is not configured.")

    temp_local_path = None
    try:
        os.makedirs(TEMP_PROCESSING_DIR, exist_ok=True)
        # Use a unique name for the temp file, incorporating original filename if desired (sanitize it)
        # For simplicity, using UUID and base of S3 key.
        base_s3_key = os.path.basename(db_image.filepath)
        temp_local_path = os.path.join(TEMP_PROCESSING_DIR, f"{uuid.uuid4().hex}_{base_s3_key}")

        logger.info(f"Attempting to download S3 object {db_image.filepath} to {temp_local_path} for face detection (image_id: {image_id})")
        storage_service.download_file(object_key=db_image.filepath, destination_path=temp_local_path)

        logger.info(f"Attempting face detection for image id {image_id} using temporary local file: {temp_local_path}")
        detected_faces_data = detect_faces(temp_local_path)

        response_faces = [FaceBoundingBox(box=f['box'], confidence=f.get('confidence')) for f in detected_faces_data]

        if not response_faces:
            logger.info(f"No faces detected for image id {image_id} (S3 key: {db_image.filepath}).")
            return FaceDetectionResponse(faces=[], image_id=db_image.id, message="No faces detected.")

        logger.info(f"Successfully detected {len(response_faces)} faces for image id {image_id} (S3 key: {db_image.filepath}).")
        return FaceDetectionResponse(faces=response_faces, image_id=db_image.id)

    except HTTPException as e: # Catch exceptions from storage_service.download_file
        logger.error(f"S3 download failed for key {db_image.filepath} (image_id: {image_id}): {e.detail}")
        # Re-raise the HTTPException (e.g., 404 if S3 object not found, 500 for other S3 errors)
        raise e
    except FileNotFoundError: # Should not happen if download_file is robust, but good for local file issues with detect_faces
        logger.error(f"Temporary local file {temp_local_path} not found after supposed download for image id {image_id}.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process image locally after S3 download.")
    except ValueError as e:
        logger.error(f"Error processing image {temp_local_path} (from S3 key {db_image.filepath}, image_id {image_id}) with face detection (ValueError): {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Invalid image file or format from S3: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during face detection for image_id {image_id} (S3 key: {db_image.filepath}): {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unexpected error during face detection.")
    finally:
        if temp_local_path and os.path.exists(temp_local_path):
            try:
                os.remove(temp_local_path)
                logger.info(f"Cleaned up temporary file: {temp_local_path}")
            except Exception as e_clean:
                logger.error(f"Failed to clean up temporary file {temp_local_path}: {e_clean}")


@app.get("/api/analysis/quality/{image_id}", response_model=ImageQualityAnalysisResponse)
@limiter.limit(get_dynamic_rate_limit)
async def get_image_quality_analysis(request: Request, image_id: uuid.UUID, db: Session = Depends(get_db)):
    db_image = crud.get_image(db, image_id=image_id)
    if not db_image:
        logger.warning(f"Image quality analysis request for non-existent image_id: {image_id}")
        raise HTTPException(status_code=404, detail=f"Image with id {image_id} not found.")

    if not db_image.filepath: # Now an S3 key
        logger.info(f"S3 key for image id {image_id} not available for quality analysis. Status: {'approved' if not db_image.rejection_reason else f'rejected ({db_image.rejection_reason})'}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"S3 key for image id {image_id} not available.")

    if not storage_service:
        logger.error("StorageService not available for image quality analysis.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Image storage service is not configured.")

    temp_local_path = None
    try:
        os.makedirs(TEMP_PROCESSING_DIR, exist_ok=True)
        base_s3_key = os.path.basename(db_image.filepath)
        temp_local_path = os.path.join(TEMP_PROCESSING_DIR, f"{uuid.uuid4().hex}_{base_s3_key}")

        logger.info(f"Attempting to download S3 object {db_image.filepath} to {temp_local_path} for quality analysis (image_id: {image_id})")
        storage_service.download_file(object_key=db_image.filepath, destination_path=temp_local_path)

        logger.info(f"Attempting image quality analysis for image id {image_id} using temporary file: {temp_local_path}")
        quality_data = analyze_image_quality(temp_local_path)

        metrics = ImageQualityMetrics(brightness=quality_data["brightness"], contrast=quality_data["contrast"])

        return ImageQualityAnalysisResponse(
            image_id=db_image.id,
            quality_metrics=metrics,
            insights=quality_data["insights"],
            message="Image quality analysis successful."
        )

    except HTTPException as e: # From storage_service.download_file
        logger.error(f"S3 download failed for key {db_image.filepath} (image_id: {image_id}) during quality analysis: {e.detail}")
        raise e
    except FileNotFoundError:
        logger.error(f"Temporary local file {temp_local_path} not found after download for image id {image_id} (quality analysis).")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process image locally after S3 download for quality analysis.")
    except ValueError as e:
        logger.error(f"Error processing image {temp_local_path} (from S3 key {db_image.filepath}, image_id {image_id}) with quality analysis (ValueError): {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Invalid image file or format from S3 for quality analysis: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during image quality analysis for image_id {image_id} (S3 key: {db_image.filepath}): {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unexpected error during image quality analysis.")
    finally:
        if temp_local_path and os.path.exists(temp_local_path):
            try:
                os.remove(temp_local_path)
                logger.info(f"Cleaned up temporary file: {temp_local_path} (quality analysis)")
            except Exception as e_clean:
                logger.error(f"Failed to clean up temporary file {temp_local_path} (quality analysis): {e_clean}")


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

    if not db_image.filepath: # S3 key
        logger.warning(f"S3 key not available for image_id: {image_id}. Status: {'approved' if not db_image.rejection_reason else f'rejected ({db_image.rejection_reason})'}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"S3 key for image id {image_id} not available.")

    if not storage_service:
        logger.error("StorageService not available for auto enhancement.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Image storage service is not configured.")

    if db_image.width is None or db_image.height is None:
        logger.error(f"Image dimensions (width, height) missing for image_id: {image_id}.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Image dimensions missing for image id {image_id}.")
    image_dimensions = (db_image.width, db_image.height)

    temp_local_path = None
    try:
        os.makedirs(TEMP_PROCESSING_DIR, exist_ok=True)
        base_s3_key = os.path.basename(db_image.filepath)
        temp_local_path = os.path.join(TEMP_PROCESSING_DIR, f"{uuid.uuid4().hex}_{base_s3_key}")

        logger.info(f"Auto-Enhance: Downloading S3 object {db_image.filepath} to {temp_local_path} for image_id: {image_id}")
        storage_service.download_file(object_key=db_image.filepath, destination_path=temp_local_path)

        face_results = []
        try:
            logger.info(f"Auto-Enhance: Performing face detection for image_id: {image_id} using {temp_local_path}")
            face_results = detect_faces(temp_local_path)
        except ValueError as e: # Bad image for face detection
            logger.error(f"Auto-Enhance: Face detection ValueError for {temp_local_path} (image_id: {image_id}): {e}")
            # Proceed with empty face_results, or raise if faces are critical for this step
        except Exception as e:
            logger.error(f"Auto-Enhance: Unexpected error during face detection for {temp_local_path} (image_id: {image_id}): {e}", exc_info=True)
            # Proceed with empty face_results

        quality_results = {}
        try:
            logger.info(f"Auto-Enhance: Performing image quality analysis for image_id: {image_id} using {temp_local_path}")
            quality_results = analyze_image_quality(temp_local_path)
        except ValueError as e: # Bad image for quality analysis
            logger.error(f"Auto-Enhance: Quality analysis ValueError for {temp_local_path} (image_id: {image_id}): {e}")
            # Proceed with empty/partial quality_results
        except Exception as e:
            logger.error(f"Auto-Enhance: Unexpected error during quality analysis for {temp_local_path} (image_id: {image_id}): {e}", exc_info=True)
            # Proceed with empty/partial quality_results

        logger.info(f"Auto-Enhance: Calculating parameters for image_id: {image_id}, mode: {mode}, using data from {temp_local_path}")
        enhancement_params_dict = calculate_auto_enhancements(
            image_path=temp_local_path, # Use downloaded temp path
            image_dimensions=image_dimensions,
            face_detection_results=face_results,
            image_quality_results=quality_results,
            mode=mode
        )
        enhancement_params_model = EnhancementParameters(**enhancement_params_dict)

        logger.info(f"Auto-Enhance: Successfully calculated parameters for image_id: {image_id}")
        return AutoEnhancementResponse(
            image_id=image_id,
            enhancement_parameters=enhancement_params_model,
            message="Auto enhancement parameters calculated successfully."
        )
    except HTTPException as e: # From storage_service.download_file
        logger.error(f"Auto-Enhance: S3 download failed for key {db_image.filepath} (image_id: {image_id}): {e.detail}")
        raise e
    except FileNotFoundError: # If temp_local_path is somehow not found after download (should not happen)
        logger.error(f"Auto-Enhance: Temp file {temp_local_path} not found after download for image_id: {image_id}.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process image locally after S3 download for auto-enhancement.")
    except Exception as e: # Catch-all for other errors, including calculate_auto_enhancements
        logger.error(f"Auto-Enhance: Error calculating parameters for image_id {image_id} (S3 key: {db_image.filepath}): {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to calculate auto enhancement parameters.")
    finally:
        if temp_local_path and os.path.exists(temp_local_path):
            try:
                os.remove(temp_local_path)
                logger.info(f"Auto-Enhance: Cleaned up temporary file: {temp_local_path}")
            except Exception as e_clean:
                logger.error(f"Auto-Enhance: Failed to clean up temporary file {temp_local_path}: {e_clean}")

@app.post("/api/enhancement/apply", response_model=ProcessedImageResponse)
@limiter.limit(get_dynamic_rate_limit)
async def apply_image_enhancements_endpoint(
    request: Request, # Changed from http_request to request for slowapi
    request_body_model: ImageEnhancementRequest, # Changed from request to request_body_model
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user) # Added current_user
):
    logger.info(f"Received request to apply enhancements for image_id: {request_body_model.image_id} by user: {current_user.id}") # Log user

    db_image = crud.get_image(db, image_id=request_body_model.image_id)
    if not db_image:
        logger.warning(f"Apply Enhancement: Image not found in DB for id: {request_body_model.image_id}")
        # Return 200 OK with error message in body as per ProcessedImageResponse model for client handling
        return ProcessedImageResponse(
            original_image_id=request_body_model.image_id,
            message="Failed to apply enhancements.",
            error=f"Image with id {request_body_model.image_id} not found."
        )

    if not db_image.filepath:
        logger.warning(f"Apply Enhancement: Filepath not available for image_id: {request_body_model.image_id}")
        return ProcessedImageResponse(
            original_image_id=request_body_model.image_id,
            message="Failed to apply enhancements.",
            error=f"S3 key for image id {request_body_model.image_id} not available."
        )

    if not storage_service: # Ensure storage_service is available earlier
        logger.error("StorageService not available for applying enhancements.")
        return ProcessedImageResponse(
            original_image_id=request_body_model.image_id,
            message="Failed to apply enhancements.",
            error="Image storage service is not configured."
        )

    temp_original_image_path = None
    processed_pil_image: Optional[PILImage.Image] = None
    try:
        # Download original image from S3 to a temporary path
        os.makedirs(TEMP_PROCESSING_DIR, exist_ok=True)
        base_s3_key_original = os.path.basename(db_image.filepath)
        temp_original_image_path = os.path.join(TEMP_PROCESSING_DIR, f"original_{uuid.uuid4().hex}_{base_s3_key_original}")

        logger.info(f"Apply Enhancement: Downloading original image {db_image.filepath} to {temp_original_image_path} for image_id: {request_body_model.image_id}")
        storage_service.download_file(object_key=db_image.filepath, destination_path=temp_original_image_path)

        face_results = []
        if request_body_model.parameters.face_smooth_intensity > 0:
            try:
                logger.info(f"Apply Enhancement: Performing face detection on {temp_original_image_path} for image_id: {request_body_model.image_id}")
                face_results = detect_faces(temp_original_image_path)
            except Exception as e_face:
                logger.error(f"Apply Enhancement: Error during face detection for {temp_original_image_path} (image_id {request_body_model.image_id}): {e_face}", exc_info=True)
                face_results = [] # Proceed without faces if detection fails

        logger.info(f"Apply Enhancement: Calling apply_enhancements service for image_id: {request_body_model.image_id} using {temp_original_image_path}")
        processed_pil_image = apply_enhancements(
            image_path=temp_original_image_path, # Use temp path of downloaded original image
            params=request_body_model.parameters.model_dump(),
            face_detection_results=face_results
        )
    except HTTPException as e_s3_download: # Catch S3 download errors specifically
        logger.error(f"Apply Enhancement: S3 download failed for {db_image.filepath} (image_id {request_body_model.image_id}): {e_s3_download.detail}")
        return ProcessedImageResponse(original_image_id=request_body_model.image_id, message="Failed to apply enhancements.", error=f"Could not retrieve original image: {e_s3_download.detail}")
    except ImageProcessingError as e:
        logger.error(f"Apply Enhancement: ImageProcessingError for image_id {request_body_model.image_id}: {e}", exc_info=True)
        return ProcessedImageResponse(
            original_image_id=request_body_model.image_id,
            message="Failed to apply enhancements due to processing error.",
            error=str(e)
        )
    except Exception as e_process: # Catch other processing errors
        logger.error(f"Apply Enhancement: Unexpected error during image processing of {temp_original_image_path} for image_id {request_body_model.image_id}: {e_process}", exc_info=True)
        return ProcessedImageResponse(
            original_image_id=request_body_model.image_id,
            message="Failed to apply enhancements due to an unexpected server error during processing.",
            error=str(e_process)
        )
    finally: # Cleanup temp original image
        if temp_original_image_path and os.path.exists(temp_original_image_path):
            try:
                os.remove(temp_original_image_path)
                logger.info(f"Apply Enhancement: Cleaned up temporary original image: {temp_original_image_path}")
            except Exception as e_clean:
                logger.error(f"Apply Enhancement: Failed to clean up temporary original image {temp_original_image_path}: {e_clean}")

    if processed_pil_image is None:
        logger.error(f"Apply Enhancement: Processing returned None for image_id: {request_body_model.image_id} (original path {db_image.filepath})")
        return ProcessedImageResponse(
            original_image_id=request_body_model.image_id,
            message="Failed to apply enhancements.",
            error="Image processing returned no result."
        )

    # Upload processed image to S3 (this part was already correct from previous steps)
    try:
        # This 'if not storage_service' check is technically redundant if the one above is hit,
        # but kept for safety within this specific try block for uploading.
        if not storage_service:
            logger.error("StorageService somehow became unavailable before S3 upload of processed image.")
            # This should ideally not be reached if the check after fetching db_image is in place.
            return ProcessedImageResponse(original_image_id=request_body_model.image_id, message="Failed to apply enhancements.", error="Storage service unavailable for upload.")

        # Convert PIL image to bytes in memory
        image_bytes_io = BytesIO()
        processed_pil_image.save(image_bytes_io, format='PNG')
        image_bytes_io.seek(0) # Reset stream position to the beginning

        # Generate S3 object key
        s3_processed_object_key = f"processed_images/{db_image.id}_enhanced_{uuid.uuid4().hex}.png"

        logger.info(f"Apply Enhancement: Uploading processed image for id {request_body_model.image_id} to S3 key: {s3_processed_object_key}")
        storage_service.upload_file(
            file_obj=image_bytes_io,
            object_key=s3_processed_object_key,
            content_type='image/png',
            acl="private"
        )

        processed_image_id_for_response = None
        db_processed_image_record = None # Initialize for wider scope

        try:
            # Create an Image record for the processed image in S3
            processed_image_filesize = image_bytes_io.getbuffer().nbytes # Get size from BytesIO
            processed_image_width, processed_image_height = processed_pil_image.size

            image_create_data = ImageCreate(
                filename=os.path.basename(s3_processed_object_key), # Use S3 key's basename as filename
                filepath=s3_processed_object_key, # S3 object key
                filesize=processed_image_filesize,
                mimetype='image/png',
                width=processed_image_width,
                height=processed_image_height,
                format='PNG'
            )
            db_processed_image_record = crud.create_image(
                db=db, image=image_create_data, width=image_create_data.width, height=image_create_data.height, format=image_create_data.format
            )
            processed_image_id_for_response = db_processed_image_record.id
            logger.info(f"Apply Enhancement: Created DB record for S3 processed image with ID: {db_processed_image_record.id}")

            # Create EnhancementHistory record
            parameters_json_str = json.dumps(request_body_model.parameters.model_dump())
            history_create_data = EnhancementHistoryBase(
                original_image_id=db_image.id,
                processed_image_id=db_processed_image_record.id,
                parameters_json=parameters_json_str
            )
            crud.create_enhancement_history(db=db, history_data=history_create_data, user_id=current_user.id)
            logger.info(f"Apply Enhancement: Enhancement history record created for user {current_user.id}, original image {db_image.id}, processed S3 image {db_processed_image_record.id}")

        except Exception as db_error:
            logger.error(f"Apply Enhancement: Error during DB operations for S3 processed image (original image id {request_body_model.image_id}): {db_error}", exc_info=True)
            # If DB ops fail, S3 object still exists. We might need a cleanup mechanism for orphaned S3 objects.
            # For now, return the S3 key or a presigned URL if possible, but flag that DB ops failed.
            # The processed_image_id_for_response will be None if create_image failed.

        # Generate presigned URL for the processed image
        presigned_url = None
        try:
            presigned_url = storage_service.generate_presigned_url(s3_processed_object_key)
        except Exception as url_gen_error:
            logger.error(f"Apply Enhancement: Failed to generate presigned URL for {s3_processed_object_key}: {url_gen_error}", exc_info=True)
            # Return S3 key as path if presigned URL fails, or handle as critical error

        return ProcessedImageResponse(
            original_image_id=request_body_model.image_id,
            processed_image_id=processed_image_id_for_response,
            processed_image_path=presigned_url or s3_processed_object_key, # Fallback to S3 key if URL gen fails
            message="Image enhancements applied and uploaded to S3 successfully." + (" DB record updated." if processed_image_id_for_response else " DB record update failed.")
        )
    except HTTPException as e: # Re-raise HTTPExceptions from storage_service.upload_file
        logger.error(f"Apply Enhancement: HTTPException during S3 upload for {request_body_model.image_id}: {e.detail}", exc_info=True)
        return ProcessedImageResponse(original_image_id=request_body_model.image_id, message="Failed to apply enhancements.", error=f"S3 upload error: {e.detail}")
    except Exception as e:
        logger.error(f"Apply Enhancement: Unexpected error during S3 upload or processing for {request_body_model.image_id}: {e}", exc_info=True)
        return ProcessedImageResponse(
            original_image_id=request_body_model.image_id,
            message="An unexpected error occurred while processing and uploading the enhanced image.",
            error=str(e)
        )


@app.post("/api/enhancement/apply-preset/{preset_id}", response_model=ProcessedImageResponse)
@limiter.limit(get_dynamic_rate_limit)
async def apply_preset_to_image_endpoint(
    request: Request, # Changed http_request to request for slowapi
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
    if not db_image.filepath: # S3 Key
        return ProcessedImageResponse(original_image_id=request_data.image_id, message="Failed to apply preset.", error=f"S3 key for image id {request_data.image_id} not available.")

    if not storage_service: # Ensure storage_service is available
        logger.error("StorageService not available for applying preset.")
        return ProcessedImageResponse(
            original_image_id=request_data.image_id,
            message="Failed to apply preset.",
            error="Image storage service is not configured."
        )

    temp_original_image_path_preset = None
    processed_pil_image: Optional[PILImage.Image] = None
    try:
        # Download original image from S3 to a temporary path
        os.makedirs(TEMP_PROCESSING_DIR, exist_ok=True)
        base_s3_key_preset_original = os.path.basename(db_image.filepath)
        temp_original_image_path_preset = os.path.join(TEMP_PROCESSING_DIR, f"preset_original_{uuid.uuid4().hex}_{base_s3_key_preset_original}")

        logger.info(f"Apply Preset: Downloading original image {db_image.filepath} to {temp_original_image_path_preset} for image_id: {request_data.image_id}")
        storage_service.download_file(object_key=db_image.filepath, destination_path=temp_original_image_path_preset)

        face_results = []
        if enhancement_params.face_smooth_intensity > 0:
            try:
                logger.info(f"Apply Preset: Performing face detection on {temp_original_image_path_preset} for image_id: {request_data.image_id}")
                face_results = detect_faces(temp_original_image_path_preset)
            except Exception as e_face_preset:
                logger.error(f"Apply Preset: Error during face detection for {temp_original_image_path_preset} (image_id {request_data.image_id}): {e_face_preset}", exc_info=True)
                face_results = []

        logger.info(f"Apply Preset: Calling apply_enhancements for image_id: {request_data.image_id} using {temp_original_image_path_preset}")
        processed_pil_image = apply_enhancements(
            image_path=temp_original_image_path_preset, # Use temp path
            params=enhancement_params.model_dump(),
            face_detection_results=face_results
        )
    except HTTPException as e_s3_download_preset:
        logger.error(f"Apply Preset: S3 download failed for {db_image.filepath} (image_id {request_data.image_id}): {e_s3_download_preset.detail}")
        return ProcessedImageResponse(original_image_id=request_data.image_id, message="Failed to apply preset.", error=f"Could not retrieve original image: {e_s3_download_preset.detail}")
    except ImageProcessingError as e_process_preset:
        logger.error(f"Apply Preset: ImageProcessingError for {temp_original_image_path_preset} (image_id {request_data.image_id}): {e_process_preset}", exc_info=True)
        return ProcessedImageResponse(original_image_id=request_data.image_id, message="Failed to apply preset due to processing error.", error=str(e_process_preset))
    except Exception as e_other_preset:
        logger.error(f"Apply Preset: Unexpected error during processing of {temp_original_image_path_preset} for image_id {request_data.image_id}: {e_other_preset}", exc_info=True)
        return ProcessedImageResponse(original_image_id=request_data.image_id, message="Failed to apply preset due to an unexpected server error.", error=str(e_other_preset))
    finally:
        if temp_original_image_path_preset and os.path.exists(temp_original_image_path_preset):
            try:
                os.remove(temp_original_image_path_preset)
                logger.info(f"Apply Preset: Cleaned up temporary original image: {temp_original_image_path_preset}")
            except Exception as e_clean_preset:
                logger.error(f"Apply Preset: Failed to clean up temporary original image {temp_original_image_path_preset}: {e_clean_preset}")

    if processed_pil_image is None:
        logger.error(f"Apply Preset: Processing returned None for image_id: {request_data.image_id} (original S3 key {db_image.filepath})")
        return ProcessedImageResponse(original_image_id=request_data.image_id, message="Failed to apply preset.", error="Image processing returned no result.")

    # 6. Save Processed Image to S3 and Record History (this part was mostly correct)
    try:
        # Redundant check, but safe
        if not storage_service:
            logger.error("StorageService became unavailable before S3 upload of preset-enhanced image.")
            return ProcessedImageResponse(original_image_id=request_data.image_id, message="Failed to apply preset.", error="Storage service unavailable for upload.")

        # Convert PIL image to bytes in memory
        image_bytes_io = BytesIO()
        processed_pil_image.save(image_bytes_io, format='PNG')
        image_bytes_io.seek(0)

        # Generate S3 object key
        s3_processed_object_key = f"processed_images/{db_image.id}_preset_enhanced_{uuid.uuid4().hex}.png"

        logger.info(f"Apply Preset: Uploading processed image for id {request_data.image_id} (preset {preset_id}) to S3 key: {s3_processed_object_key}")
        storage_service.upload_file(
            file_obj=image_bytes_io,
            object_key=s3_processed_object_key,
            content_type='image/png',
            acl="private"
        )

        processed_image_id_for_response = None
        db_processed_image_record = None

        try:
            processed_image_filesize = image_bytes_io.getbuffer().nbytes
            processed_image_width, processed_image_height = processed_pil_image.size
            img_create_data = ImageCreate(
                filename=os.path.basename(s3_processed_object_key),
                filepath=s3_processed_object_key, # S3 object key
                filesize=processed_image_filesize,
                mimetype='image/png',
                width=processed_image_width,
                height=processed_image_height,
                format='PNG'
            )
            db_processed_image_record = crud.create_image(db=db, image=img_create_data, width=img_create_data.width, height=img_create_data.height, format=img_create_data.format)
            processed_image_id_for_response = db_processed_image_record.id
            logger.info(f"Apply Preset: Created DB record for S3 processed image {db_processed_image_record.id}")

            history_params_dict = enhancement_params.model_dump()
            history_params_dict["applied_preset_id"] = str(preset_id)
            parameters_json_for_history = json.dumps(history_params_dict)
            history_create_data = EnhancementHistoryBase(
                original_image_id=db_image.id,
                processed_image_id=db_processed_image_record.id,
                parameters_json=parameters_json_for_history
            )
            crud.create_enhancement_history(db=db, history_data=history_create_data, user_id=current_user.id)
            logger.info(f"Apply Preset: Enhancement history created for user {current_user.id}, original image {db_image.id}, processed S3 {db_processed_image_record.id}, preset {preset_id}")

        except Exception as db_error:
            logger.error(f"Apply Preset: DB error for original image {request_data.image_id}, preset {preset_id}: {db_error}", exc_info=True)
            # S3 object still exists. Consider cleanup or logging for orphaned objects.

        presigned_url = None
        try:
            presigned_url = storage_service.generate_presigned_url(s3_processed_object_key)
        except Exception as url_gen_error:
            logger.error(f"Apply Preset: Failed to generate presigned URL for {s3_processed_object_key}: {url_gen_error}", exc_info=True)

        return ProcessedImageResponse(
            original_image_id=request_data.image_id,
            processed_image_id=processed_image_id_for_response,
            processed_image_path=presigned_url or s3_processed_object_key, # Fallback to key
            message="Preset applied and image uploaded to S3 successfully." + (" DB record updated." if processed_image_id_for_response else " DB record update failed.")
        )
    except HTTPException as e: # Re-raise from storage_service
        logger.error(f"Apply Preset: HTTPException during S3 upload for {request_data.image_id}, preset {preset_id}: {e.detail}", exc_info=True)
        return ProcessedImageResponse(original_image_id=request_data.image_id, message="Failed to apply preset.", error=f"S3 upload error: {e.detail}")
    except Exception as e:
        logger.error(f"Apply Preset: Unexpected error during S3 upload or processing for {request_data.image_id}, preset {preset_id}: {e}", exc_info=True)
        return ProcessedImageResponse(original_image_id=request_data.image_id, message="An unexpected error occurred while applying preset and uploading.", error=str(e))
