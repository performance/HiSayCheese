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

# --- Unit Tests for Content Moderation ---
from unittest.mock import MagicMock, patch
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
