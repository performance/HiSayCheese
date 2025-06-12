# services/content_moderation.py
import io
import logging
from google.cloud import vision
from PIL import Image as PILImage
from schemas.analysis_schemas import ContentModerationResult # Import from your new schema file

# Content moderation function using Google Cloud Vision API
# Schemas are now in a separate directory

from config import CONTENT_MODERATION_TYPE

logger = logging.getLogger(__name__)


async def moderate_image_content(contents: bytes) -> ContentModerationResult:
    """
    Switches between real Google Vision moderation and a local mock
    based on the CONTENT_MODERATION_TYPE environment variable.
    """
    if CONTENT_MODERATION_TYPE == "local":
        logger.info("Content moderation is in 'local' mode. Approving image automatically.")
        return ContentModerationResult(is_approved=True, rejection_reason=None)
    
    # If not local, run the real moderation
    return await _moderate_with_google_vision(contents)


async def _moderate_with_google_vision(contents: bytes) -> ContentModerationResult:
    """The original function that connects to Google Cloud Vision."""
    # This will only be called if CONTENT_MODERATION_TYPE is not 'local'
    try:
        from google.cloud import vision
    except ImportError:
        logger.error("google-cloud-vision is not installed. Cannot perform real content moderation.")
        return ContentModerationResult(is_approved=False, rejection_reason="Moderation service not configured.")

    """
    Moderates image content using Google Cloud Vision API.
    - Checks for portraits using face detection.
    - Checks for prohibited content using safe search detection.
    """
    try:
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

