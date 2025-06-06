import os
import shutil
import uuid
from pathlib import Path

import pytest
from fastapi import UploadFile, File, Depends, HTTPException, status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

# Adjust the import according to your project structure for 'app'
# Assuming 'app' is directly in 'main.py' at the root level alongside 'db', 'models', 'tests' folders
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app, UPLOAD_DIR, MAX_FILE_SIZE_BYTES, ALLOWED_MIME_TYPES, MIME_TYPE_TO_EXTENSION
from db import crud
from models import models
from db.database import create_db_and_tables, SessionLocal, engine, get_db # Assuming engine is not used directly here for now

# --- Test Setup & Fixtures ---

client = TestClient(app)

# Minimal valid file headers (approximate for testing, python-magic might be stricter)
MINIMAL_JPG_CONTENT = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x15\x14\x11\x14\x18\x16\x12\x18\x15\x1c\x1e\x1d\x1a\x1c\x18\x1a\x15\x14\xff\xc9\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\xd2\xcf\x20\xff\xd9"
MINIMAL_PNG_CONTENT = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
MINIMAL_WEBP_CONTENT = b"RIFF\x24\x00\x00\x00WEBPVP8 \x10\x00\x00\x00\x90\x01\x00\x9d\x01\x2a\x01\x00\x01\x00\x02\x00\x34\x25\xa4\x00\x03\x70\x00\xfe\xfb\xf2\x80\x00\x00"

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    # This ensures the main DB schema is created if it doesn't exist.
    # For truly isolated tests, a separate test DB (e.g., in-memory) and dedicated
    # engine/sessionmaker would be used, and app.dependency_overrides for get_db.
    # For this setup, we rely on function-scoped fixtures to clean data.
    create_db_and_tables()
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    yield
    # Optional: cleanup after all tests in a session
    # shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    # db_file = str(engine.url).split("///")[-1] # try to get db file name
    # if os.path.exists(db_file) and "mem" not in db_file:
    #     os.remove(db_file)


@pytest.fixture(scope="function", autouse=True)
def cleanup_upload_dir_and_db_records():
    # Clear UPLOAD_DIR before each test and recreate
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Clear Image table records before each test
    db = SessionLocal()
    try:
        db.query(models.Image).delete()
        # If other tables are affected by uploads or need clearing, do it here.
        # e.g., db.query(models.Number).delete()
        db.commit()
    except Exception as e:
        db.rollback()
        pytest.fail(f"DB cleanup failed: {e}")
    finally:
        db.close()
    yield # Test runs here


def create_dummy_file_for_upload(filename: str, content: bytes, content_type: str):
    from io import BytesIO
    return (filename, BytesIO(content), content_type)

# --- Test Cases ---

def test_upload_valid_jpg():
    file_to_upload = create_dummy_file_for_upload("test.jpg", MINIMAL_JPG_CONTENT, "image/jpeg")
    response = client.post("/api/images/upload", files={"file": file_to_upload})

    assert response.status_code == status.HTTP_201_CREATED, response.text
    data = response.json()
    assert "id" in data
    assert data["filename"] == "test.jpg"
    assert data["mimetype"] == "image/jpeg" # This should be the magic-detected one
    assert UPLOAD_DIR in data["filepath"]
    assert data["filepath"].endswith(".jpg") # Based on MIME_TYPE_TO_EXTENSION

    assert os.path.exists(data["filepath"])

    db = SessionLocal()
    try:
        img_id = uuid.UUID(data["id"])
        db_img = db.query(models.Image).filter(models.Image.id == img_id).first()
        assert db_img is not None
        assert db_img.filename == "test.jpg"
        assert db_img.mimetype == "image/jpeg"
    finally:
        db.close()

def test_upload_valid_png():
    file_to_upload = create_dummy_file_for_upload("test.png", MINIMAL_PNG_CONTENT, "image/png")
    response = client.post("/api/images/upload", files={"file": file_to_upload})

    assert response.status_code == status.HTTP_201_CREATED, response.text
    data = response.json()
    assert data["mimetype"] == "image/png"
    assert data["filepath"].endswith(".png")
    assert os.path.exists(data["filepath"])

def test_upload_valid_webp():
    file_to_upload = create_dummy_file_for_upload("test.webp", MINIMAL_WEBP_CONTENT, "image/webp")
    response = client.post("/api/images/upload", files={"file": file_to_upload})

    assert response.status_code == status.HTTP_201_CREATED, response.text
    data = response.json()
    assert data["mimetype"] == "image/webp"
    assert data["filepath"].endswith(".webp")
    assert os.path.exists(data["filepath"])

def test_upload_pdf_unsupported_type():
    pdf_content = b"%PDF-1.4 fake content. This is definitely not an image."
    file_to_upload = create_dummy_file_for_upload("test.pdf", pdf_content, "application/pdf")
    response = client.post("/api/images/upload", files={"file": file_to_upload})
    assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, response.text

def test_upload_too_large():
    large_content = b"a" * (MAX_FILE_SIZE_BYTES + 1)
    file_to_upload = create_dummy_file_for_upload("large.jpg", large_content, "image/jpeg") # Content type doesn't matter as size check is first
    response = client.post("/api/images/upload", files={"file": file_to_upload})
    assert response.status_code == status.HTTP_413_PAYLOAD_TOO_LARGE, response.text

def test_upload_corrupted_image_as_jpg_extension_but_text_content():
    corrupted_content = b"This is just plain text, not a jpg."
    # Client claims it's image/jpeg due to filename, but python-magic will detect text/plain.
    file_to_upload = create_dummy_file_for_upload("corrupted.jpg", corrupted_content, "image/jpeg")
    response = client.post("/api/images/upload", files={"file": file_to_upload})
    assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, response.text
    # The detail message should indicate the detected type if the main app includes it.
    # e.g. "Unsupported file type: 'text/plain'. Allowed types are JPG, PNG, WEBP."
    assert "Unsupported file type" in response.json()["detail"]
    assert "text/plain" in response.json()["detail"] # Check if detected type is mentioned

def test_unique_filename_generation():
    file1_to_upload = create_dummy_file_for_upload("same_name.jpg", MINIMAL_JPG_CONTENT, "image/jpeg")
    file2_to_upload = create_dummy_file_for_upload("same_name.jpg", MINIMAL_JPG_CONTENT, "image/jpeg")

    response1 = client.post("/api/images/upload", files={"file": file1_to_upload})
    assert response1.status_code == status.HTTP_201_CREATED, response1.text
    data1 = response1.json()

    response2 = client.post("/api/images/upload", files={"file": file2_to_upload})
    assert response2.status_code == status.HTTP_201_CREATED, response2.text
    data2 = response2.json()

    assert data1["filepath"] != data2["filepath"]

    filename1_on_server = os.path.basename(data1["filepath"])
    filename2_on_server = os.path.basename(data2["filepath"])

    assert len(filename1_on_server.split('.')[0]) == 32
    assert len(filename2_on_server.split('.')[0]) == 32

    try:
        uuid.UUID(filename1_on_server.split('.')[0], version=4)
        uuid.UUID(filename2_on_server.split('.')[0], version=4)
    except ValueError:
        pytest.fail(f"Filename base {filename1_on_server.split('.')[0]} or {filename2_on_server.split('.')[0]} is not a valid UUID hex.")

    assert os.path.exists(data1["filepath"])
    assert os.path.exists(data2["filepath"])

def test_health_check_get_number_not_set():
    # Assuming the number is not set by default after cleanup
    response = client.get("/health")
    assert response.status_code == status.HTTP_404_NOT_FOUND # If no number is set
    assert response.json() == {"message": "No number set yet"}

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"message": "Hello World"}

# Self-check print statements (removed in final version, useful for debugging setup)
# print(f"Test UPLOAD_DIR: {os.path.abspath(UPLOAD_DIR)}")
# try:
#     import magic
#     print("python-magic is available in test environment.")
# except ImportError:
#     print("CRITICAL: python-magic is NOT available in test environment.")
#     pytest.fail("python-magic import failed in test file")

import io # Added io
from typing import Optional # Added Optional
from PIL import Image as PILImage # Added PILImage for creating test images
from PIL import ExifTags # Added ExifTags for creating exif data

# --- Unit Tests for Content Moderation ---
from unittest.mock import MagicMock, patch, mock_open
# Assuming ContentModerationResult is accessible from main or models.models
# If main.ContentModerationResult is pydantic model imported from pydantic
# and models.ContentModerationResult is something else, ensure correct import.
# Based on previous subtasks, ContentModerationResult is defined in main.py
from main import moderate_image_content, ContentModerationResult
from google.cloud import vision # For vision.Likelihood constants

# Using the existing MINIMAL_WEBP_CONTENT for test image bytes
# If a different one is needed, define it here. For mocking, content isn't deeply inspected by mocks.
MINIMAL_WEBP_BYTES_FOR_MODERATION = MINIMAL_WEBP_CONTENT # Reuse if suitable

@pytest.mark.asyncio # Mark test as async
async def test_moderate_content_approved():
    """Image is a clear portrait and has no prohibited content."""
    mock_image_pil = MagicMock()
    mock_image_pil.size = (1000, 800) # width, height

    mock_vision_client_instance = MagicMock()

    # Mock face_detection response
    mock_face_response = MagicMock()
    mock_face = MagicMock()
    mock_face.detection_confidence = 0.95
    # Large face covering > 5% of 1000x800=800000 area. (600*600 = 360000, which is > 40000)
    vertices = [MagicMock(x=100, y=100), MagicMock(x=700, y=100), MagicMock(x=700, y=700), MagicMock(x=100, y=700)]
    # Ensure the vertices attribute is correctly assigned to bounding_poly
    mock_face.bounding_poly.vertices = vertices
    mock_face_response.face_annotations = [mock_face]
    mock_face_response.error.message = ""
    mock_vision_client_instance.face_detection.return_value = mock_face_response

    # Mock safe_search_detection response
    mock_safe_response = MagicMock()
    mock_safe_annotation = MagicMock() # Create a separate mock for the annotation attribute
    mock_safe_annotation.adult = vision.Likelihood.VERY_UNLIKELY
    mock_safe_annotation.violence = vision.Likelihood.VERY_UNLIKELY
    mock_safe_annotation.racy = vision.Likelihood.VERY_UNLIKELY
    mock_safe_response.safe_search_annotation = mock_safe_annotation # Assign to the attribute
    mock_safe_response.error.message = ""
    mock_vision_client_instance.safe_search_detection.return_value = mock_safe_response

    with patch('main.PILImage.open', return_value=mock_image_pil) as mock_pil_open, \
         patch('main.vision.ImageAnnotatorClient', return_value=mock_vision_client_instance) as mock_vision_client_constructor:

        result = await moderate_image_content(MINIMAL_WEBP_BYTES_FOR_MODERATION)

        assert result.is_approved is True
        assert result.rejection_reason is None
        mock_pil_open.assert_called_once_with(pytest.ANY) # Check PILImage.open was called
        mock_vision_client_constructor.assert_called_once() # Check client was initialized
        mock_vision_client_instance.face_detection.assert_called_once()
        mock_vision_client_instance.safe_search_detection.assert_called_once()

@pytest.mark.asyncio
async def test_moderate_content_no_face():
    """Image has no face detected."""
    mock_image_pil = MagicMock()
    mock_image_pil.size = (1000, 800)

    mock_vision_client_instance = MagicMock()
    mock_face_response = MagicMock()
    mock_face_response.face_annotations = [] # No faces
    mock_face_response.error.message = ""
    mock_vision_client_instance.face_detection.return_value = mock_face_response

    with patch('main.PILImage.open', return_value=mock_image_pil) as mock_pil_open, \
         patch('main.vision.ImageAnnotatorClient', return_value=mock_vision_client_instance) as mock_vision_client_constructor:

        result = await moderate_image_content(MINIMAL_WEBP_BYTES_FOR_MODERATION)

        assert result.is_approved is False
        assert result.rejection_reason == "No clear portrait found or face is not prominent."
        mock_pil_open.assert_called_once()
        mock_vision_client_constructor.assert_called_once()
        mock_vision_client_instance.face_detection.assert_called_once()
        mock_vision_client_instance.safe_search_detection.assert_not_called()

@pytest.mark.asyncio
async def test_moderate_content_face_too_small():
    """Face detected but too small."""
    mock_image_pil = MagicMock()
    mock_image_pil.size = (1000, 800)

    mock_vision_client_instance = MagicMock()
    mock_face_response = MagicMock()
    mock_face = MagicMock()
    mock_face.detection_confidence = 0.95
    # Small face: 10x10 = 100 area. 5% of 800000 is 40000. This is < 5%.
    vertices = [MagicMock(x=0, y=0), MagicMock(x=10, y=0), MagicMock(x=10, y=10), MagicMock(x=0, y=10)]
    mock_face.bounding_poly.vertices = vertices
    mock_face_response.face_annotations = [mock_face]
    mock_face_response.error.message = ""
    mock_vision_client_instance.face_detection.return_value = mock_face_response

    with patch('main.PILImage.open', return_value=mock_image_pil) as mock_pil_open, \
         patch('main.vision.ImageAnnotatorClient', return_value=mock_vision_client_instance) as mock_vision_client_constructor:

        result = await moderate_image_content(MINIMAL_WEBP_BYTES_FOR_MODERATION)

        assert result.is_approved is False
        assert result.rejection_reason == "No clear portrait found or face is not prominent."
        mock_vision_client_instance.face_detection.assert_called_once()
        mock_vision_client_instance.safe_search_detection.assert_not_called()

@pytest.mark.asyncio
async def test_moderate_content_prohibited_adult():
    """Portrait is fine, but adult content detected."""
    mock_image_pil = MagicMock()
    mock_image_pil.size = (1000, 800)

    mock_vision_client_instance = MagicMock()
    mock_face_response = MagicMock()
    mock_face = MagicMock()
    mock_face.detection_confidence = 0.95
    vertices = [MagicMock(x=100, y=100), MagicMock(x=700, y=100), MagicMock(x=700, y=700), MagicMock(x=100, y=700)]
    mock_face.bounding_poly.vertices = vertices
    mock_face_response.face_annotations = [mock_face]
    mock_face_response.error.message = ""
    mock_vision_client_instance.face_detection.return_value = mock_face_response

    mock_safe_response = MagicMock()
    mock_safe_annotation = MagicMock()
    mock_safe_annotation.adult = vision.Likelihood.LIKELY # Prohibited
    mock_safe_annotation.violence = vision.Likelihood.VERY_UNLIKELY
    mock_safe_annotation.racy = vision.Likelihood.VERY_UNLIKELY
    mock_safe_response.safe_search_annotation = mock_safe_annotation
    mock_safe_response.error.message = ""
    mock_vision_client_instance.safe_search_detection.return_value = mock_safe_response

    with patch('main.PILImage.open', return_value=mock_image_pil) as mock_pil_open, \
         patch('main.vision.ImageAnnotatorClient', return_value=mock_vision_client_instance) as mock_vision_client_constructor:

        result = await moderate_image_content(MINIMAL_WEBP_BYTES_FOR_MODERATION)

        assert result.is_approved is False
        assert "adult" in result.rejection_reason.lower() # Check specific reason if possible
        assert result.rejection_reason == "Image content violates guidelines (adult, violence, or racy content detected)."
        mock_vision_client_instance.face_detection.assert_called_once()
        mock_vision_client_instance.safe_search_detection.assert_called_once()

@pytest.mark.asyncio
async def test_moderate_content_vision_face_detection_api_error():
    """Vision API returns an error during face detection."""
    mock_image_pil = MagicMock()
    mock_image_pil.size = (1000, 800)

    mock_vision_client_instance = MagicMock()
    mock_face_response = MagicMock()
    mock_face_response.face_annotations = []
    mock_face_response.error.message = "API internal error" # Error message from API
    mock_vision_client_instance.face_detection.return_value = mock_face_response

    with patch('main.PILImage.open', return_value=mock_image_pil) as mock_pil_open, \
         patch('main.vision.ImageAnnotatorClient', return_value=mock_vision_client_instance) as mock_vision_client_constructor:

        result = await moderate_image_content(MINIMAL_WEBP_BYTES_FOR_MODERATION)

        assert result.is_approved is False
        assert result.rejection_reason == "Face detection failed: API internal error"
        mock_vision_client_instance.face_detection.assert_called_once()
        mock_vision_client_instance.safe_search_detection.assert_not_called()

@pytest.mark.asyncio
async def test_moderate_content_vision_safe_search_api_error():
    """Vision API returns an error during safe search."""
    mock_image_pil = MagicMock()
    mock_image_pil.size = (1000, 800)

    mock_vision_client_instance = MagicMock()
    mock_face_response = MagicMock()
    mock_face = MagicMock()
    mock_face.detection_confidence = 0.95
    vertices = [MagicMock(x=100, y=100), MagicMock(x=700, y=100), MagicMock(x=700, y=700), MagicMock(x=100, y=700)]
    mock_face.bounding_poly.vertices = vertices
    mock_face_response.face_annotations = [mock_face]
    mock_face_response.error.message = ""
    mock_vision_client_instance.face_detection.return_value = mock_face_response

    mock_safe_response = MagicMock()
    # No safe_search_annotation if API itself errors out before populating it
    mock_safe_response.error.message = "SafeSearch API internal error"
    mock_vision_client_instance.safe_search_detection.return_value = mock_safe_response

    with patch('main.PILImage.open', return_value=mock_image_pil) as mock_pil_open, \
         patch('main.vision.ImageAnnotatorClient', return_value=mock_vision_client_instance) as mock_vision_client_constructor:

        result = await moderate_image_content(MINIMAL_WEBP_BYTES_FOR_MODERATION)

        assert result.is_approved is False
        assert result.rejection_reason == "SafeSearch detection failed: SafeSearch API internal error"
        mock_vision_client_instance.face_detection.assert_called_once()
        mock_vision_client_instance.safe_search_detection.assert_called_once()

@pytest.mark.asyncio
async def test_moderate_content_pillow_error():
    """Pillow fails to open the image."""
    with patch('main.PILImage.open', side_effect=Exception("Pillow can't open this")) as mock_pil_open, \
         patch('main.vision.ImageAnnotatorClient') as mock_vision_client_constructor: # Mock client so it's not called

        result = await moderate_image_content(MINIMAL_WEBP_BYTES_FOR_MODERATION)

        assert result.is_approved is False
        assert result.rejection_reason == "Failed to process image properties."
        mock_pil_open.assert_called_once()
        mock_vision_client_constructor.assert_called_once() # Client is initialized before Pillow error in current code
        # Depending on exact structure, client might not be called if Pillow fails early.
        # The current code initializes client, then tries Pillow.
        # mock_vision_client_constructor.return_value.face_detection.assert_not_called()
        # mock_vision_client_constructor.return_value.safe_search_detection.assert_not_called()
        # Let's check the mock_vision_client_instance calls if constructor was called
        if mock_vision_client_constructor.called:
            mock_vision_client_instance = mock_vision_client_constructor.return_value
            mock_vision_client_instance.face_detection.assert_not_called()
            mock_vision_client_instance.safe_search_detection.assert_not_called()


# --- Helper functions for creating dummy image files ---
def create_dummy_image_bytes(
    width: int,
    height: int,
    img_format: str = "JPEG",
    exif_dict: Optional[dict] = None
) -> bytes:
    """
    Creates dummy image bytes with specified properties.
    exif_dict: A dictionary where keys are EXIF tag IDs and values are the tag values.
               Example: {0x0112: 3} for orientation.
    """
    img_byte_arr = io.BytesIO()
    image = PILImage.new("RGB", (width, height), color="blue")

    exif_bytes = b""
    if exif_dict:
        exif = PILImage.Exif()
        for tag, value in exif_dict.items():
            exif[tag] = value
        exif_bytes = exif.tobytes()

    if img_format.upper() == "JPEG":
        # Ensure exif data is only passed for JPEG and if it exists
        if exif_bytes:
            image.save(img_byte_arr, format="JPEG", exif=exif_bytes)
        else:
            image.save(img_byte_arr, format="JPEG")
    elif img_format.upper() == "PNG":
        image.save(img_byte_arr, format="PNG")
    else:
        raise ValueError(f"Unsupported image format for dummy creation: {img_format}")

    return img_byte_arr.getvalue()

# --- Fixtures for Metadata Tests ---

@pytest.fixture
def mock_moderate_content_approve_fixture(mocker):
    """Mocks moderate_image_content to always return approved."""
    # This mock will be applied to 'main.moderate_image_content'
    return mocker.patch(
        "main.moderate_image_content",
        return_value=ContentModerationResult(is_approved=True, rejection_reason=None)
    )

# --- Security Headers Test ---
def test_security_headers_present():
    response = client.get("/") # Any endpoint should have these headers
    assert response.status_code == 200
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert "default-src 'self'" in response.headers.get("Content-Security-Policy")
    assert "script-src 'self'" in response.headers.get("Content-Security-Policy")
    assert "object-src 'none'" in response.headers.get("Content-Security-Policy")
    assert "frame-ancestors 'none'" in response.headers.get("Content-Security-Policy")
    # HSTS is often only sent over HTTPS, TestClient uses HTTP by default.
    # If app is configured to force HTTPS or TestClient to use HTTPS, this can be asserted.
    # For now, we'll assume it might not be present in basic HTTP test environment.
    # assert "max-age=31536000" in response.headers.get("Strict-Transport-Security", "")


# --- Request Body Size Limit Test ---
# Using /put_number endpoint for testing general JSON payload size limit
# MAX_REQUEST_BODY_SIZE is 1MB in main.py
# MAX_FILE_SIZE_BYTES is for file uploads (larger)

# Need to get MAX_REQUEST_BODY_SIZE from main.py for the test
from main import MAX_REQUEST_BODY_SIZE as APP_MAX_REQUEST_BODY_SIZE

def test_request_body_too_large_for_json_endpoint():
    # Create a payload slightly larger than MAX_REQUEST_BODY_SIZE
    # The /put_number endpoint expects {"value": int}
    # We'll send a large string for "value" to make the JSON large,
    # though the endpoint will fail validation (422) if it gets that far.
    # The middleware should intercept it with 413 before Pydantic validation.
    # However, a simple way is to make the key itself large or many keys.
    # Let's try making a large number of key-value pairs.
    large_payload_dict = {}
    # Approximate size: each pair "keyX": 0, is about 10 bytes.
    # So, for 1MB, we need about 100,000 pairs.
    num_pairs = (APP_MAX_REQUEST_BODY_SIZE // 10) + 100 # Ensure it's over
    for i in range(num_pairs):
        large_payload_dict[f"key{i}"] = i

    # This test assumes that the /put_number endpoint is NOT excluded by the middleware.
    # The middleware in main.py currently does not exclude /put_number.
    response = client.post("/put_number", json=large_payload_dict)
    assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, response.text

# --- File Upload Sanitization Tests ---
@pytest.mark.usefixtures("mock_moderate_content_approve_fixture", "mock_file_system_operations_fixture")
def test_upload_malicious_filename_path_traversal(client):
    # Ensure the mock for os.path.join in mock_file_system_operations_fixture
    # reflects the sanitized name if we want to assert the saved path.
    # The current mock_file_system_operations_fixture uses a static "mock_saved_file.jpg".
    # We might need a more dynamic mock or to inspect the 'image_data_to_create'
    # that gets passed to crud.create_image. For now, let's focus on the filename in the DB.

    malicious_filename = "../../../etc/passwd"
    # Expected sanitized: "etc_passwd" or similar, depending on secure_filename
    # from werkzeug.utils import secure_filename as werkzeug_secure_filename
    # expected_sanitized_name_by_werkzeug = werkzeug_secure_filename(malicious_filename)
    # print(f"Werkzeug sanitized: {expected_sanitized_name_by_werkzeug}") -> becomes "etc_passwd"

    file_to_upload = create_dummy_file_for_upload(malicious_filename, MINIMAL_JPG_CONTENT, "image/jpeg")

    response = client.post("/api/images/upload", files={"file": file_to_upload})
    assert response.status_code == status.HTTP_201_CREATED, response.text
    data = response.json()

    # Check the filename stored in the response/DB (should be sanitized)
    # The 'filename' field in ImageCreate is what we're interested in.
    # secure_filename changes "../../../etc/passwd" to "etc_passwd"
    assert data["filename"] == "etc_passwd", f"Filename was {data['filename']}, expected etc_passwd"

    # Also ensure the server_filepath in the DB record (if accessible here) is safe
    # The mock_file_system_operations_fixture currently returns a fixed path.
    # To properly test this, we'd need to inspect the arguments to os.path.join
    # or ensure the DB record has the sanitized name in its path.
    db = SessionLocal()
    try:
        img_id = uuid.UUID(data["id"])
        db_img = db.query(models.Image).filter(models.Image.id == img_id).first()
        assert db_img is not None
        assert db_img.filename == "etc_passwd" # Check DB record
        # The filepath stored in DB should also use the sanitized name components.
        # Our current UPLOAD_DIR + unique_hex + extension structure means the original
        # malicious filename isn't directly part of the server path construction,
        # only the final extension is taken from the original (sanitized) mime type.
        # The `image_data_to_create.filename` is what gets saved in the DB model.
        # The actual file path on server is `UPLOAD_DIR/uuid.hex.actual_extension`.
        # So, the main check is that `db_img.filename` (which comes from `file.filename` after sanitization) is safe.
    finally:
        db.close()

def test_upload_malicious_filename_script_tag(client):
    malicious_filename = "<script>alert('evil')</script>.jpg"
    # secure_filename changes this to "script_alert_evil_script.jpg"
    expected_sanitized = "script_alert_evil_script.jpg"

    file_to_upload = create_dummy_file_for_upload(malicious_filename, MINIMAL_JPG_CONTENT, "image/jpeg")
    response = client.post("/api/images/upload", files={"file": file_to_upload})
    assert response.status_code == status.HTTP_201_CREATED, response.text
    data = response.json()
    assert data["filename"] == expected_sanitized


# --- Pydantic Validation Tests for a Generic Endpoint (e.g., /put_number) ---
def test_put_number_missing_value():
    response = client.post("/put_number", json={}) # Missing 'value'
    assert response.status_code == 422, response.text
    data = response.json()
    assert any(err["loc"] == ["body", "value"] and "Missing" in err["msg"] for err in data["detail"])

def test_put_number_invalid_type():
    response = client.post("/put_number", json={"value": "not-an-integer"})
    assert response.status_code == 422, response.text
    data = response.json()
    assert any(err["loc"] == ["body", "value"] and "Input should be a valid integer" in err["msg"] for err in data["detail"])

def test_put_number_out_of_range_negative():
    # Assuming NumberCreate.value has conint(ge=0) from previous model updates
    response = client.post("/put_number", json={"value": -10})
    assert response.status_code == 422, response.text
    data = response.json()
    assert any(err["loc"] == ["body", "value"] and "greater than or equal to 0" in err["msg"] for err in data["detail"])


# Malicious string inputs for a generic JSON field (e.g., ImageCreate.rejection_reason if it were settable via API)
# For /put_number, the 'value' is an int, so not suitable for string attacks.
# Let's consider the /api/images/upload endpoint and its string fields in ImageCreate,
# although these are not directly set by user JSON but derived from file properties.
# The `filename` is already tested for sanitization.
# `mimetype`, `format`, `color_profile` are also string-based but derived.
# The one place a user *might* inject strings that get stored is if an error message
# from an external service (like Vision API mock) was directly put into `rejection_reason`.
# However, our current `moderate_image_content` returns fixed strings or from `error.message`.
# This test is more relevant for endpoints that take arbitrary user JSON with string fields.
# For now, we'll skip direct SQLi/XSS tests on string fields in `test_main.py` as `upload_image`
# doesn't take arbitrary JSON strings that are directly stored without sanitization of the field itself.
# Filename sanitization is the key one for `upload_image`.

@pytest.fixture
def mock_file_system_operations_fixture(mocker):
    """Mocks file system operations like open, makedirs, path.join for uploads."""
    mocker.patch("os.makedirs") # Mock os.makedirs
    # Mock builtins.open for writing the file, allow read for other parts of app if necessary
    # For the upload, we are interested in the 'wb' mode.
    mock_file_open = mock_open()
    mocker.patch("builtins.open", mock_file_open)

    # Mock os.path.join to control the returned filepath string for assertions
    # The actual value might need to be dynamic or based on input filename if tests need it
    # For now, a static path is fine as long as it's consistent.
    # UPLOAD_DIR is "uploads/images/"
    mock_join = mocker.patch("os.path.join", return_value=f"{UPLOAD_DIR.rstrip('/')}/mock_saved_file.jpg")

    return {
        "open": mock_file_open,
        "makedirs": os.makedirs, # Access the original mock if needed for assertions
        "join": mock_join
    }


# --- Test Cases for Image Upload Metadata ---

@pytest.mark.usefixtures("mock_moderate_content_approve_fixture", "mock_file_system_operations_fixture")
class TestImageUploadMetadata:

    def test_upload_jpeg_metadata(self, client):
        img_bytes = create_dummy_image_bytes(width=120, height=80, img_format="JPEG")
        filesize = len(img_bytes)

        response = client.post(
            "/api/images/upload",
            files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")}
        )

        assert response.status_code == status.HTTP_201_CREATED, response.text
        data = response.json()

        assert data["filename"] == "test.jpg"
        assert data["mimetype"] == "image/jpeg"
        assert data["filesize"] == filesize
        assert data["width"] == 120
        assert data["height"] == 80
        assert data["format"] == "JPEG"
        assert data["exif_orientation"] is None # Default JPEG from Pillow might not have it
        assert data["color_profile"] == "RGB" # Pillow default RGB
        assert data["rejection_reason"] is None
        assert UPLOAD_DIR in data["filepath"] # Path comes from mock_file_system_operations_fixture

    def test_upload_png_metadata(self, client):
        img_bytes = create_dummy_image_bytes(width=100, height=60, img_format="PNG")
        filesize = len(img_bytes)

        response = client.post(
            "/api/images/upload",
            files={"file": ("test.png", io.BytesIO(img_bytes), "image/png")}
        )

        assert response.status_code == status.HTTP_201_CREATED, response.text
        data = response.json()

        assert data["filename"] == "test.png"
        assert data["mimetype"] == "image/png"
        assert data["filesize"] == filesize
        assert data["width"] == 100
        assert data["height"] == 60
        assert data["format"] == "PNG"
        assert data["exif_orientation"] is None # PNGs typically don't have EXIF
        assert data["color_profile"] == "RGB" # Pillow default RGB for PNG
        assert data["rejection_reason"] is None
        assert UPLOAD_DIR in data["filepath"]

    def test_upload_jpeg_with_exif_orientation(self, client):
        orientation_tag_id = 0x0112 # Orientation
        orientation_value = 3 # Rotate 180 degrees
        img_bytes = create_dummy_image_bytes(
            width=150, height=100, img_format="JPEG",
            exif_dict={orientation_tag_id: orientation_value}
        )
        filesize = len(img_bytes)

        response = client.post(
            "/api/images/upload",
            files={"file": ("exif_test.jpg", io.BytesIO(img_bytes), "image/jpeg")}
        )

        assert response.status_code == status.HTTP_201_CREATED, response.text
        data = response.json()

        assert data["width"] == 150
        assert data["height"] == 100
        assert data["format"] == "JPEG"
        assert data["filesize"] == filesize
        assert data["exif_orientation"] == orientation_value
        assert data["color_profile"] == "RGB"
        assert data["rejection_reason"] is None

    def test_upload_jpeg_no_exif_data(self, client):
        # create_dummy_image_bytes by default creates JPEG with no EXIF unless specified
        img_bytes = create_dummy_image_bytes(width=80, height=50, img_format="JPEG")
        filesize = len(img_bytes)

        response = client.post(
            "/api/images/upload",
            files={"file": ("no_exif.jpg", io.BytesIO(img_bytes), "image/jpeg")}
        )

        assert response.status_code == status.HTTP_201_CREATED, response.text
        data = response.json()

        assert data["width"] == 80
        assert data["height"] == 50
        assert data["format"] == "JPEG"
        assert data["filesize"] == filesize
        assert data["exif_orientation"] is None
        assert data["color_profile"] == "RGB"
        assert data["rejection_reason"] is None

    def test_upload_jpeg_minimal_exif_no_orientation(self, client):
        # Provide some other EXIF tag, but not orientation
        other_exif_tag_id = 0x010F # Make (Camera Manufacturer)
        other_exif_value = "TestCam"
        img_bytes = create_dummy_image_bytes(
            width=70, height=40, img_format="JPEG",
            exif_dict={other_exif_tag_id: other_exif_value}
        )
        filesize = len(img_bytes)

        response = client.post(
            "/api/images/upload",
            files={"file": ("minimal_exif.jpg", io.BytesIO(img_bytes), "image/jpeg")}
        )

        assert response.status_code == status.HTTP_201_CREATED, response.text
        data = response.json()

        assert data["width"] == 70
        assert data["height"] == 40
        assert data["format"] == "JPEG"
        assert data["filesize"] == filesize
        assert data["exif_orientation"] is None # Orientation tag was not included
        assert data["color_profile"] == "RGB"
        assert data["rejection_reason"] is None
        # Check if other EXIF might have been unintentionally parsed (it shouldn't be by current main.py logic)
        # This test primarily ensures exif_orientation is None.


@pytest.mark.asyncio
async def test_moderate_content_general_exception_during_api_call():
    """A general (non-API error message) exception occurs during Vision API interaction."""
    mock_image_pil = MagicMock()
    mock_image_pil.size = (1000, 800)

    mock_vision_client_instance = MagicMock()
    # Simulate an error when face_detection is called, e.g. network issue
    mock_vision_client_instance.face_detection.side_effect = Exception("Network timeout")

    with patch('main.PILImage.open', return_value=mock_image_pil) as mock_pil_open, \
         patch('main.vision.ImageAnnotatorClient', return_value=mock_vision_client_instance) as mock_vision_client_constructor:

        result = await moderate_image_content(MINIMAL_WEBP_BYTES_FOR_MODERATION)

        assert result.is_approved is False
        assert result.rejection_reason == "Content moderation check failed due to an internal error."
        mock_pil_open.assert_called_once()
        mock_vision_client_constructor.assert_called_once()
        mock_vision_client_instance.face_detection.assert_called_once() # Attempted
        mock_vision_client_instance.safe_search_detection.assert_not_called() # Should not be reached
```
