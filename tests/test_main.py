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

from main import app, MAX_FILE_SIZE_BYTES, ALLOWED_MIME_TYPES, MIME_TYPE_TO_EXTENSION, TEMP_PROCESSING_DIR
# UPLOAD_DIR is removed from main.py, so remove from here too.
from db import crud
from models import models
from db.database import create_db_and_tables, SessionLocal, engine, get_db

# Moto for S3 mocking
from moto import mock_s3
import boto3 # To interact with moto's mock S3
from botocore.exceptions import ClientError # Import ClientError for S3 checks
from unittest.mock import patch # Added for patching S3 upload failures

# --- Test Setup & Fixtures ---

client = TestClient(app)

# Minimal valid file headers (approximate for testing, python-magic might be stricter)
MINIMAL_JPG_CONTENT = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x15\x14\x11\x14\x18\x16\x12\x18\x15\x1c\x1e\x1d\x1a\x1c\x18\x1a\x15\x14\xff\xc9\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\xd2\xcf\x20\xff\xd9"
MINIMAL_PNG_CONTENT = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
MINIMAL_WEBP_CONTENT = b"RIFF\x24\x00\x00\x00WEBPVP8 \x10\x00\x00\x00\x90\x01\x00\x9d\x01\x2a\x01\x00\x01\x00\x02\x00\x34\x25\xa4\x00\x03\x70\x00\xfe\xfb\xf2\x80\x00\x00"

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    create_db_and_tables()
    os.makedirs(TEMP_PROCESSING_DIR, exist_ok=True)

    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_S3_REGION"] = "us-east-1"
    os.environ["AWS_S3_BUCKET_NAME"] = "test-bucket"

    yield
    shutil.rmtree(TEMP_PROCESSING_DIR, ignore_errors=True)
    del os.environ["AWS_ACCESS_KEY_ID"]
    del os.environ["AWS_SECRET_ACCESS_KEY"]
    del os.environ["AWS_SESSION_TOKEN"]
    del os.environ["AWS_S3_REGION"]
    del os.environ["AWS_S3_BUCKET_NAME"]


@pytest.fixture(scope="function", autouse=True)
def cleanup_s3_and_db_records():
    shutil.rmtree(TEMP_PROCESSING_DIR, ignore_errors=True)
    os.makedirs(TEMP_PROCESSING_DIR, exist_ok=True)

    db = SessionLocal()
    try:
        db.query(models.Image).delete()
        db.query(models.UserPreset).delete()
        db.query(models.EnhancementHistory).delete()
        db.commit()
    except Exception as e:
        db.rollback()
        pytest.fail(f"DB cleanup failed: {e}")
    finally:
        db.close()
    yield


def create_dummy_file_for_upload(filename: str, content: bytes, content_type: str):
    from io import BytesIO
    return (filename, BytesIO(content), content_type)

# --- Test Cases ---

@mock_s3
def test_upload_valid_jpg():
    s3 = boto3.client("s3", region_name=os.environ["AWS_S3_REGION"])
    try:
        s3.create_bucket(Bucket=os.environ["AWS_S3_BUCKET_NAME"])
    except ClientError as e:
        if e.response['Error']['Code'] != 'BucketAlreadyOwnedByYou':
            raise

    file_to_upload = create_dummy_file_for_upload("test.jpg", MINIMAL_JPG_CONTENT, "image/jpeg")
    response = client.post("/api/images/upload", files={"file": file_to_upload})

    assert response.status_code == status.HTTP_201_CREATED, response.text
    data = response.json()
    assert "id" in data
    assert data["filename"] == "test.jpg"
    assert data["mimetype"] == "image/jpeg"
    assert data["filepath"].startswith("original_images/")
    assert data["filepath"].endswith(".jpg")
    assert "presigned_url" in data
    assert data["presigned_url"] is not None
    assert "https://test-bucket.s3.us-east-1.amazonaws.com/original_images/" in data["presigned_url"]

    try:
        s3_object = s3.get_object(Bucket=os.environ["AWS_S3_BUCKET_NAME"], Key=data["filepath"])
        assert s3_object is not None
        assert s3_object['ContentLength'] == len(MINIMAL_JPG_CONTENT)
        assert s3_object['ContentType'] == "image/jpeg"
    except ClientError as e:
        pytest.fail(f"S3 get_object failed for {data['filepath']}: {e}")

    db = SessionLocal()
    try:
        img_id = uuid.UUID(data["id"])
        db_img = db.query(models.Image).filter(models.Image.id == img_id).first()
        assert db_img is not None
        assert db_img.filename == "test.jpg"
        assert db_img.mimetype == "image/jpeg"
        assert db_img.filepath == data["filepath"]
    finally:
        db.close()

@mock_s3
def test_upload_valid_png():
    s3 = boto3.client("s3", region_name=os.environ["AWS_S3_REGION"])
    s3.create_bucket(Bucket=os.environ["AWS_S3_BUCKET_NAME"])

    file_to_upload = create_dummy_file_for_upload("test.png", MINIMAL_PNG_CONTENT, "image/png")
    response = client.post("/api/images/upload", files={"file": file_to_upload})

    assert response.status_code == status.HTTP_201_CREATED, response.text
    data = response.json()
    assert data["mimetype"] == "image/png"
    assert data["filepath"].startswith("original_images/")
    assert data["filepath"].endswith(".png")
    assert data["presigned_url"] is not None
    s3.get_object(Bucket=os.environ["AWS_S3_BUCKET_NAME"], Key=data["filepath"])

@mock_s3
def test_upload_valid_webp():
    s3 = boto3.client("s3", region_name=os.environ["AWS_S3_REGION"])
    s3.create_bucket(Bucket=os.environ["AWS_S3_BUCKET_NAME"])

    file_to_upload = create_dummy_file_for_upload("test.webp", MINIMAL_WEBP_CONTENT, "image/webp")
    response = client.post("/api/images/upload", files={"file": file_to_upload})

    assert response.status_code == status.HTTP_201_CREATED, response.text
    data = response.json()
    assert data["mimetype"] == "image/webp"
    assert data["filepath"].startswith("original_images/")
    assert data["filepath"].endswith(".webp")
    assert data["presigned_url"] is not None
    s3.get_object(Bucket=os.environ["AWS_S3_BUCKET_NAME"], Key=data["filepath"])

@mock_s3
def test_upload_s3_upload_failure(client):
    s3 = boto3.client("s3", region_name=os.environ["AWS_S3_REGION"])
    s3.create_bucket(Bucket=os.environ["AWS_S3_BUCKET_NAME"])
    file_to_upload = create_dummy_file_for_upload("test_s3_fail.jpg", MINIMAL_JPG_CONTENT, "image/jpeg")
    with patch("main.storage_service.upload_file", side_effect=ClientError({"Error": {"Code": "InternalError", "Message": "S3 is down"}}, "PutObject")) as mock_upload:
        response = client.post("/api/images/upload", files={"file": file_to_upload})
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to upload file to S3" in response.json()["detail"] or "S3 is down" in response.json()["detail"]
        mock_upload.assert_called_once()
    db = SessionLocal()
    try:
        count_before = db.query(models.Image).count()
        file_to_upload_2 = create_dummy_file_for_upload("test_s3_fail_db_check.jpg", MINIMAL_JPG_CONTENT, "image/jpeg")
        with patch("main.storage_service.upload_file", side_effect=ClientError({"Error": {"Code": "InternalError", "Message": "S3 is down"}}, "PutObject")):
            client.post("/api/images/upload", files={"file": file_to_upload_2})
        count_after = db.query(models.Image).count()
        assert count_after == count_before, "DB record should not be created if S3 upload fails before DB stage"
    finally:
        db.close()

@mock_s3
def test_upload_pdf_unsupported_type():
    pdf_content = b"%PDF-1.4 fake content. This is definitely not an image."
    file_to_upload = create_dummy_file_for_upload("test.pdf", pdf_content, "application/pdf")
    response = client.post("/api/images/upload", files={"file": file_to_upload})
    assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, response.text

def test_upload_too_large():
    large_content = b"a" * (MAX_FILE_SIZE_BYTES + 1)
    file_to_upload = create_dummy_file_for_upload("large.jpg", large_content, "image/jpeg")
    response = client.post("/api/images/upload", files={"file": file_to_upload})
    assert response.status_code == status.HTTP_413_PAYLOAD_TOO_LARGE, response.text

def test_upload_corrupted_image_as_jpg_extension_but_text_content():
    corrupted_content = b"This is just plain text, not a jpg."
    file_to_upload = create_dummy_file_for_upload("corrupted.jpg", corrupted_content, "image/jpeg")
    response = client.post("/api/images/upload", files={"file": file_to_upload})
    assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, response.text
    assert "Unsupported file type" in response.json()["detail"]
    assert "text/plain" in response.json()["detail"]

@mock_s3
def test_unique_filename_generation():
    s3 = boto3.client("s3", region_name=os.environ["AWS_S3_REGION"])
    s3.create_bucket(Bucket=os.environ["AWS_S3_BUCKET_NAME"])
    file1_to_upload = create_dummy_file_for_upload("test.jpg", MINIMAL_JPG_CONTENT, "image/jpeg")
    file2_to_upload = create_dummy_file_for_upload("another_name.jpg", MINIMAL_JPG_CONTENT, "image/jpeg")

    response1 = client.post("/api/images/upload", files={"file": file1_to_upload})
    assert response1.status_code == status.HTTP_201_CREATED, response1.text
    data1 = response1.json()

    response2 = client.post("/api/images/upload", files={"file": file2_to_upload})
    assert response2.status_code == status.HTTP_201_CREATED, response2.text
    data2 = response2.json()

    assert data1["filepath"] != data2["filepath"]
    assert data1["filepath"].startswith("original_images/")
    assert data2["filepath"].startswith("original_images/")
    s3.head_object(Bucket=os.environ["AWS_S3_BUCKET_NAME"], Key=data1["filepath"])
    s3.head_object(Bucket=os.environ["AWS_S3_BUCKET_NAME"], Key=data2["filepath"])

def test_health_check_get_number_not_set():
    response = client.get("/health")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json() == {"message": "No number set yet"}

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"message": "Hello World"}

import io
from typing import Optional
from PIL import Image as PILImage
from PIL import ExifTags
from unittest.mock import MagicMock, patch, mock_open
from main import moderate_image_content, ContentModerationResult
from google.cloud import vision

MINIMAL_WEBP_BYTES_FOR_MODERATION = MINIMAL_WEBP_CONTENT

@pytest.mark.asyncio
async def test_moderate_content_approved():
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
    mock_safe_annotation.adult = vision.Likelihood.VERY_UNLIKELY
    mock_safe_annotation.violence = vision.Likelihood.VERY_UNLIKELY
    mock_safe_annotation.racy = vision.Likelihood.VERY_UNLIKELY
    mock_safe_response.safe_search_annotation = mock_safe_annotation
    mock_safe_response.error.message = ""
    mock_vision_client_instance.safe_search_detection.return_value = mock_safe_response
    with patch('main.PILImage.open', return_value=mock_image_pil) as mock_pil_open, \
         patch('main.vision.ImageAnnotatorClient', return_value=mock_vision_client_instance) as mock_vision_client_constructor:
        result = await moderate_image_content(MINIMAL_WEBP_BYTES_FOR_MODERATION)
        assert result.is_approved is True
        assert result.rejection_reason is None
        mock_pil_open.assert_called_once_with(pytest.ANY)
        mock_vision_client_constructor.assert_called_once()
        mock_vision_client_instance.face_detection.assert_called_once()
        mock_vision_client_instance.safe_search_detection.assert_called_once()

@pytest.mark.asyncio
async def test_moderate_content_no_face():
    mock_image_pil = MagicMock()
    mock_image_pil.size = (1000, 800)
    mock_vision_client_instance = MagicMock()
    mock_face_response = MagicMock()
    mock_face_response.face_annotations = []
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
    mock_image_pil = MagicMock()
    mock_image_pil.size = (1000, 800)
    mock_vision_client_instance = MagicMock()
    mock_face_response = MagicMock()
    mock_face = MagicMock()
    mock_face.detection_confidence = 0.95
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
    mock_safe_annotation.adult = vision.Likelihood.LIKELY
    mock_safe_annotation.violence = vision.Likelihood.VERY_UNLIKELY
    mock_safe_annotation.racy = vision.Likelihood.VERY_UNLIKELY
    mock_safe_response.safe_search_annotation = mock_safe_annotation
    mock_safe_response.error.message = ""
    mock_vision_client_instance.safe_search_detection.return_value = mock_safe_response
    with patch('main.PILImage.open', return_value=mock_image_pil) as mock_pil_open, \
         patch('main.vision.ImageAnnotatorClient', return_value=mock_vision_client_instance) as mock_vision_client_constructor:
        result = await moderate_image_content(MINIMAL_WEBP_BYTES_FOR_MODERATION)
        assert result.is_approved is False
        assert "adult" in result.rejection_reason.lower()
        assert result.rejection_reason == "Image content violates guidelines (adult, violence, or racy content detected)."
        mock_vision_client_instance.face_detection.assert_called_once()
        mock_vision_client_instance.safe_search_detection.assert_called_once()

@pytest.mark.asyncio
async def test_moderate_content_vision_face_detection_api_error():
    mock_image_pil = MagicMock()
    mock_image_pil.size = (1000, 800)
    mock_vision_client_instance = MagicMock()
    mock_face_response = MagicMock()
    mock_face_response.face_annotations = []
    mock_face_response.error.message = "API internal error"
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
    with patch('main.PILImage.open', side_effect=Exception("Pillow can't open this")) as mock_pil_open, \
         patch('main.vision.ImageAnnotatorClient') as mock_vision_client_constructor:
        result = await moderate_image_content(MINIMAL_WEBP_BYTES_FOR_MODERATION)
        assert result.is_approved is False
        assert result.rejection_reason == "Failed to process image properties."
        mock_pil_open.assert_called_once()
        mock_vision_client_constructor.assert_called_once()
        if mock_vision_client_constructor.called:
            mock_vision_client_instance = mock_vision_client_constructor.return_value
            mock_vision_client_instance.face_detection.assert_not_called()
            mock_vision_client_instance.safe_search_detection.assert_not_called()

def create_dummy_image_bytes(
    width: int,
    height: int,
    img_format: str = "JPEG",
    exif_dict: Optional[dict] = None
) -> bytes:
    img_byte_arr = io.BytesIO()
    image = PILImage.new("RGB", (width, height), color="blue")
    exif_bytes = b""
    if exif_dict:
        exif = PILImage.Exif()
        for tag, value in exif_dict.items():
            exif[tag] = value
        exif_bytes = exif.tobytes()
    if img_format.upper() == "JPEG":
        if exif_bytes:
            image.save(img_byte_arr, format="JPEG", exif=exif_bytes)
        else:
            image.save(img_byte_arr, format="JPEG")
    elif img_format.upper() == "PNG":
        image.save(img_byte_arr, format="PNG")
    else:
        raise ValueError(f"Unsupported image format for dummy creation: {img_format}")
    return img_byte_arr.getvalue()

@pytest.fixture
def mock_moderate_content_approve_fixture(mocker):
    return mocker.patch(
        "main.moderate_image_content",
        return_value=ContentModerationResult(is_approved=True, rejection_reason=None)
    )

TEST_ORIGIN = "http://example.com"

def test_cors_basic_get_request_with_origin(client: TestClient):
    response = client.get("/", headers={"Origin": TEST_ORIGIN})
    assert response.status_code == 200
    assert response.headers.get("Access-Control-Allow-Origin") == "*"
    assert response.headers.get("Access-Control-Allow-Credentials") == "true"

def test_cors_preflight_options_request(client: TestClient):
    request_headers = {
        "Origin": TEST_ORIGIN,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type, Authorization",
    }
    response = client.options("/", headers=request_headers)
    assert response.status_code == 200
    assert response.headers.get("Access-Control-Allow-Origin") == TEST_ORIGIN
    allowed_methods = response.headers.get("Access-Control-Allow-Methods")
    assert allowed_methods is not None
    if "*" not in allowed_methods:
        assert "POST" in allowed_methods.upper()
    allowed_headers = response.headers.get("Access-Control-Allow-Headers")
    assert allowed_headers is not None
    if "*" not in allowed_headers:
        requested_headers = set(h.strip().lower() for h in "Content-Type, Authorization".lower().split(','))
        returned_allowed_headers = set(h.strip().lower() for h in allowed_headers.split(','))
        assert requested_headers.issubset(returned_allowed_headers)
    assert response.headers.get("Access-Control-Allow-Credentials") == "true"

import time

try:
    from main import ANON_USER_RATE_LIMIT, AUTH_USER_RATE_LIMIT
    ANON_REQUESTS_PER_WINDOW = int(ANON_USER_RATE_LIMIT.split('/')[0])
    AUTH_REQUESTS_PER_WINDOW = int(AUTH_USER_RATE_LIMIT.split('/')[0])
except ImportError:
    ANON_REQUESTS_PER_WINDOW = 20
    AUTH_REQUESTS_PER_WINDOW = 100

def get_rate_limit_headers_from_response(response):
    return {
        "limit": response.headers.get("X-RateLimit-Limit"),
        "remaining": response.headers.get("X-RateLimit-Remaining"),
        "reset": response.headers.get("X-RateLimit-Reset"),
    }

def create_user_and_get_token(client_instance, db_session_instance, email_prefix="auth_test_user"):
    user_email = f"{email_prefix}_{uuid.uuid4()}@example.com"
    user_password = "ValidPasswordForTesting1!"
    reg_response = client_instance.post(
        "/api/auth/register",
        json={"email": user_email, "password": user_password},
    )
    if reg_response.status_code != status.HTTP_201_CREATED:
        db = SessionLocal()
        existing_user = db.query(models.User).filter(models.User.email == user_email).first()
        if existing_user:
            db.delete(existing_user)
            db.commit()
        db.close()
        reg_response = client_instance.post(
            "/api/auth/register", json={"email": user_email, "password": user_password}
        )
    assert reg_response.status_code == status.HTTP_201_CREATED, \
        f"Failed to register user for token generation: {reg_response.text}"
    login_response = client_instance.post(
        "/api/auth/login",
        data={"username": user_email, "password": user_password},
    )
    assert login_response.status_code == status.HTTP_200_OK, \
        f"Failed to login user for token generation: {login_response.text}"
    token = login_response.json()["access_token"]
    return token, user_email

def clear_user_from_db(db: Session, email: str):
    user = db.query(models.User).filter(models.User.email == email).first()
    if user:
        db.delete(user)
        db.commit()

def test_security_headers_present():
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert "default-src 'self'" in response.headers.get("Content-Security-Policy")
    assert "script-src 'self'" in response.headers.get("Content-Security-Policy")
    assert "object-src 'none'" in response.headers.get("Content-Security-Policy")
    assert "frame-ancestors 'none'" in response.headers.get("Content-Security-Policy")

from main import MAX_REQUEST_BODY_SIZE as APP_MAX_REQUEST_BODY_SIZE

def test_request_body_too_large_for_json_endpoint():
    large_payload_dict = {}
    num_pairs = (APP_MAX_REQUEST_BODY_SIZE // 10) + 100
    for i in range(num_pairs):
        large_payload_dict[f"key{i}"] = i
    response = client.post("/put_number", json=large_payload_dict)
    assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, response.text

@pytest.mark.usefixtures("mock_moderate_content_approve_fixture")
@mock_s3
def test_upload_malicious_filename_path_traversal(client):
    s3 = boto3.client("s3", region_name=os.environ["AWS_S3_REGION"])
    s3.create_bucket(Bucket=os.environ["AWS_S3_BUCKET_NAME"])
    malicious_filename = "../../../etc/passwd"
    file_to_upload = create_dummy_file_for_upload(malicious_filename, MINIMAL_JPG_CONTENT, "image/jpeg")
    response = client.post("/api/images/upload", files={"file": file_to_upload})
    assert response.status_code == status.HTTP_201_CREATED, response.text
    data = response.json()
    assert data["filename"] == "etc_passwd", f"Filename was {data['filename']}, expected etc_passwd"
    db = SessionLocal()
    try:
        img_id = uuid.UUID(data["id"])
        db_img = db.query(models.Image).filter(models.Image.id == img_id).first()
        assert db_img is not None
        assert db_img.filename == "etc_passwd"
    finally:
        db.close()

@pytest.mark.usefixtures("mock_moderate_content_approve_fixture")
@mock_s3
def test_upload_malicious_filename_script_tag(client):
    s3 = boto3.client("s3", region_name=os.environ["AWS_S3_REGION"])
    s3.create_bucket(Bucket=os.environ["AWS_S3_BUCKET_NAME"])
    malicious_filename = "<script>alert('evil')</script>.jpg"
    expected_sanitized = "script_alert_evil_script.jpg"
    file_to_upload = create_dummy_file_for_upload(malicious_filename, MINIMAL_JPG_CONTENT, "image/jpeg")
    response = client.post("/api/images/upload", files={"file": file_to_upload})
    assert response.status_code == status.HTTP_201_CREATED, response.text
    data = response.json()
    assert data["filename"] == expected_sanitized

def test_put_number_missing_value():
    response = client.post("/put_number", json={})
    assert response.status_code == 422, response.text
    data = response.json()
    assert any(err["loc"] == ["body", "value"] and "Missing" in err["msg"] for err in data["detail"])

def test_put_number_invalid_type():
    response = client.post("/put_number", json={"value": "not-an-integer"})
    assert response.status_code == 422, response.text
    data = response.json()
    assert any(err["loc"] == ["body", "value"] and "Input should be a valid integer" in err["msg"] for err in data["detail"])

def test_put_number_out_of_range_negative():
    response = client.post("/put_number", json={"value": -10})
    assert response.status_code == 422, response.text
    data = response.json()
    assert any(err["loc"] == ["body", "value"] and "greater than or equal to 0" in err["msg"] for err in data["detail"])

@pytest.mark.usefixtures("mock_moderate_content_approve_fixture")
@mock_s3
def test_upload_image_rate_limiting_anonymous(client):
    s3 = boto3.client("s3", region_name=os.environ["AWS_S3_REGION"])
    s3.create_bucket(Bucket=os.environ["AWS_S3_BUCKET_NAME"])
    for i in range(ANON_REQUESTS_PER_WINDOW):
        file_to_upload = create_dummy_file_for_upload(f"anon_upload_{i}.jpg", MINIMAL_JPG_CONTENT, "image/jpeg")
        response = client.post("/api/images/upload", files={"file": file_to_upload})
        if response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            pytest.fail(f"Anonymous upload attempt {i+1} rate limited prematurely.")
        assert response.status_code == status.HTTP_201_CREATED
    file_to_upload = create_dummy_file_for_upload("anon_upload_final.jpg", MINIMAL_JPG_CONTENT, "image/jpeg")
    response = client.post("/api/images/upload", files={"file": file_to_upload})
    assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    headers = get_rate_limit_headers_from_response(response)
    assert headers["limit"] == str(ANON_REQUESTS_PER_WINDOW)
    assert headers["remaining"] == "0"

@pytest.mark.usefixtures("mock_moderate_content_approve_fixture")
@mock_s3
def test_upload_image_rate_limiting_authenticated(client, db_session):
    s3 = boto3.client("s3", region_name=os.environ["AWS_S3_REGION"])
    s3.create_bucket(Bucket=os.environ["AWS_S3_BUCKET_NAME"])
    token, user_email_for_cleanup = create_user_and_get_token(client, db_session, "upload_auth_rl")
    auth_headers = {"Authorization": f"Bearer {token}"}
    for i in range(AUTH_REQUESTS_PER_WINDOW):
        file_to_upload = create_dummy_file_for_upload(f"auth_upload_{i}.jpg", MINIMAL_JPG_CONTENT, "image/jpeg")
        response = client.post("/api/images/upload", files={"file": file_to_upload}, headers=auth_headers)
        if response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            pytest.fail(f"Authenticated upload attempt {i+1} rate limited prematurely.")
        assert response.status_code == status.HTTP_201_CREATED
    file_to_upload = create_dummy_file_for_upload("auth_upload_final.jpg", MINIMAL_JPG_CONTENT, "image/jpeg")
    response = client.post("/api/images/upload", files={"file": file_to_upload}, headers=auth_headers)
    assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    headers = get_rate_limit_headers_from_response(response)
    assert headers["limit"] == str(AUTH_REQUESTS_PER_WINDOW)
    assert headers["remaining"] == "0"
    db = SessionLocal()
    clear_user_from_db(db, user_email_for_cleanup)
    db.close()

@pytest.mark.usefixtures("mock_moderate_content_approve_fixture")
@mock_s3
def test_upload_image_rate_limit_headers_anonymous_single_request(client):
    s3 = boto3.client("s3", region_name=os.environ["AWS_S3_REGION"])
    s3.create_bucket(Bucket=os.environ["AWS_S3_BUCKET_NAME"])
    file_to_upload = create_dummy_file_for_upload("header_test_anon.jpg", MINIMAL_JPG_CONTENT, "image/jpeg")
    response = client.post("/api/images/upload", files={"file": file_to_upload})
    assert response.status_code == status.HTTP_201_CREATED
    headers = get_rate_limit_headers_from_response(response)
    assert headers["limit"] == str(ANON_REQUESTS_PER_WINDOW)
    assert headers["remaining"] == str(ANON_REQUESTS_PER_WINDOW - 1)
    assert headers["reset"] is not None
    assert int(headers["reset"]) > time.time() - 5

@pytest.mark.usefixtures("mock_moderate_content_approve_fixture")
@mock_s3
def test_upload_image_rate_limit_headers_authenticated_single_request(client, db_session):
    s3 = boto3.client("s3", region_name=os.environ["AWS_S3_REGION"])
    s3.create_bucket(Bucket=os.environ["AWS_S3_BUCKET_NAME"])
    token, user_email_for_cleanup = create_user_and_get_token(client, db_session, "upload_hdr_auth_rl")
    auth_headers = {"Authorization": f"Bearer {token}"}
    file_to_upload = create_dummy_file_for_upload("header_test_auth.jpg", MINIMAL_JPG_CONTENT, "image/jpeg")
    response = client.post("/api/images/upload", files={"file": file_to_upload}, headers=auth_headers)
    assert response.status_code == status.HTTP_201_CREATED
    headers = get_rate_limit_headers_from_response(response)
    assert headers["limit"] == str(AUTH_REQUESTS_PER_WINDOW)
    assert headers["remaining"] == str(AUTH_REQUESTS_PER_WINDOW - 1)
    assert headers["reset"] is not None
    assert int(headers["reset"]) > time.time() - 5
    db = SessionLocal()
    clear_user_from_db(db, user_email_for_cleanup)
    db.close()

@pytest.mark.usefixtures("mock_moderate_content_approve_fixture")
@mock_s3
def test_upload_image_rate_limit_reset_anonymous(client):
    s3 = boto3.client("s3", region_name=os.environ["AWS_S3_REGION"])
    s3.create_bucket(Bucket=os.environ["AWS_S3_BUCKET_NAME"])
    for i in range(ANON_REQUESTS_PER_WINDOW + 1):
        file_to_upload = create_dummy_file_for_upload(f"reset_anon_upload_{i}.jpg", MINIMAL_JPG_CONTENT, "image/jpeg")
        response = client.post("/api/images/upload", files={"file": file_to_upload})
        if i == ANON_REQUESTS_PER_WINDOW:
            assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
            headers_429 = get_rate_limit_headers_from_response(response)
            reset_time = int(headers_429["reset"])
            current_time = int(time.time())
            sleep_duration = max(0, reset_time - current_time) + 1
            if sleep_duration > 65:
                pytest.skip(f"Reset time too far ({sleep_duration}s), skipping sleep.")
            time.sleep(sleep_duration)
    file_to_upload = create_dummy_file_for_upload("reset_anon_upload_final.jpg", MINIMAL_JPG_CONTENT, "image/jpeg")
    response_after_reset = client.post("/api/images/upload", files={"file": file_to_upload})
    assert response_after_reset.status_code == status.HTTP_201_CREATED
    headers_after = get_rate_limit_headers_from_response(response_after_reset)
    assert headers_after["remaining"] == str(ANON_REQUESTS_PER_WINDOW - 1)

@pytest.fixture(scope="function")
def uploaded_image_id(client):
    s3_client = boto3.client("s3", region_name=os.environ.get("AWS_S3_REGION", "us-east-1"))
    bucket_name = os.environ.get("AWS_S3_BUCKET_NAME", "test-bucket")
    try:
        s3_client.create_bucket(Bucket=bucket_name)
    except ClientError as e:
        if e.response['Error']['Code'] != 'BucketAlreadyOwnedByYou' and e.response['Error']['Code'] != 'BucketAlreadyExists':
             raise
    img_bytes = create_dummy_image_bytes(10,10, img_format="JPEG")
    file_data = ("setup_image_for_enh.jpg", io.BytesIO(img_bytes), "image/jpeg")
    with patch("main.moderate_image_content", return_value=ContentModerationResult(is_approved=True, rejection_reason=None)):
        response = client.post("/api/images/upload", files={"file": file_data})
    assert response.status_code == status.HTTP_201_CREATED, f"Setup S3 image upload failed: {response.text}"
    data = response.json()
    return {"id": data["id"], "s3_key": data["filepath"]}

@pytest.fixture(scope="function")
def s3_image_for_enhancement(client):
    s3_client = boto3.client("s3", region_name=os.environ.get("AWS_S3_REGION", "us-east-1"))
    bucket_name = os.environ.get("AWS_S3_BUCKET_NAME", "test-bucket")
    try:
        s3_client.create_bucket(Bucket=bucket_name)
    except ClientError as e:
        if e.response['Error']['Code'] != 'BucketAlreadyOwnedByYou' and e.response['Error']['Code'] != 'BucketAlreadyExists':
            raise
    img_bytes = create_dummy_image_bytes(width=100, height=100, img_format="JPEG")
    files = {"file": ("test_for_enh.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    with patch("main.moderate_image_content", return_value=ContentModerationResult(is_approved=True)):
        response = client.post("/api/images/upload", files=files)
    assert response.status_code == status.HTTP_201_CREATED
    response_data = response.json()
    return {"id": response_data["id"], "s3_key": response_data["filepath"]}

@mock_s3
def test_analysis_faces_endpoint_rate_limited_anon(client, uploaded_image_id):
    endpoint_url = f"/api/analysis/faces/{uploaded_image_id['id']}"
    for i in range(ANON_REQUESTS_PER_WINDOW):
        response = client.get(endpoint_url)
        if response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            pytest.fail(f"Anonymous request {i+1} to {endpoint_url} rate limited prematurely.")
        assert response.status_code != status.HTTP_429_TOO_MANY_REQUESTS
    response = client.get(endpoint_url)
    assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS

@mock_s3
def test_enhancement_apply_endpoint_rate_limited_auth(client, uploaded_image_id, db_session):
    token, user_email_for_cleanup = create_user_and_get_token(client, db_session, "enh_apply_rl")
    auth_headers = {"Authorization": f"Bearer {token}"}
    endpoint_url = "/api/enhancement/apply"
    enhancement_params = {
        "brightness_target": 1.1, "contrast_target": 1.1, "saturation_target": 1.1,
        "background_blur_radius": 0, "crop_rect": [0,0,10,10], "face_smooth_intensity": 0.0
    }
    request_body = {"image_id": uploaded_image_id['id'], "parameters": enhancement_params}
    for i in range(AUTH_REQUESTS_PER_WINDOW):
        response = client.post(endpoint_url, json=request_body, headers=auth_headers)
        if response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            pytest.fail(f"Authenticated request {i+1} to {endpoint_url} rate limited prematurely.")
        assert response.status_code != status.HTTP_429_TOO_MANY_REQUESTS
    response = client.post(endpoint_url, json=request_body, headers=auth_headers)
    assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    db = SessionLocal()
    clear_user_from_db(db, user_email_for_cleanup)
    db.close()

@mock_s3
def test_apply_enhancement_success(client, s3_image_for_enhancement, db_session):
    image_id = s3_image_for_enhancement["id"]
    original_s3_key = s3_image_for_enhancement["s3_key"]
    s3 = boto3.client("s3", region_name=os.environ["AWS_S3_REGION"])
    bucket_name = os.environ["AWS_S3_BUCKET_NAME"]
    s3.head_object(Bucket=bucket_name, Key=original_s3_key)
    token, user_email = create_user_and_get_token(client, db_session, "enh_apply_succ")
    auth_headers = {"Authorization": f"Bearer {token}"}
    enhancement_params = {
        "brightness_target": 1.2, "contrast_target": 1.1, "saturation_target": 1.0,
        "background_blur_radius": 0, "crop_rect": [0,0,100,100], "face_smooth_intensity": 0.0
    }
    request_body = {"image_id": image_id, "parameters": enhancement_params}
    response = client.post("/api/enhancement/apply", json=request_body, headers=auth_headers)
    assert response.status_code == status.HTTP_200_OK, response.text
    data = response.json()
    assert data["original_image_id"] == image_id
    assert data["processed_image_id"] is not None
    assert data["processed_image_path"] is not None
    assert "https://test-bucket.s3.us-east-1.amazonaws.com/processed_images/" in data["processed_image_path"]
    assert data["error"] is None
    db = SessionLocal()
    try:
        processed_img_record = db.query(models.Image).filter(models.Image.id == data["processed_image_id"]).first()
        assert processed_img_record is not None
        assert processed_img_record.filepath.startswith("processed_images/")
        s3.head_object(Bucket=bucket_name, Key=processed_img_record.filepath)
    finally:
        db.close()
    clear_user_from_db(SessionLocal(), user_email)

@mock_s3
def test_apply_enhancement_s3_original_download_failure(client, s3_image_for_enhancement, db_session):
    image_id = s3_image_for_enhancement["id"]
    token, user_email = create_user_and_get_token(client, db_session, "enh_dl_fail")
    auth_headers = {"Authorization": f"Bearer {token}"}
    enhancement_params = {"brightness_target": 1.1, "contrast_target": 1.1, "saturation_target": 1.1, "background_blur_radius": 0, "crop_rect": [0,0,10,10], "face_smooth_intensity": 0.0}
    request_body = {"image_id": image_id, "parameters": enhancement_params}
    with patch("main.storage_service.download_file", side_effect=HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Simulated S3 Original Not Found")) as mock_download:
        response = client.post("/api/enhancement/apply", json=request_body, headers=auth_headers)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["original_image_id"] == image_id
    assert data["processed_image_id"] is None
    assert data["processed_image_path"] is None
    assert "Could not retrieve original image" in data["error"] or "Simulated S3 Original Not Found" in data["error"]
    mock_download.assert_called_once()
    clear_user_from_db(SessionLocal(), user_email)

@mock_s3
def test_apply_enhancement_s3_processed_upload_failure(client, s3_image_for_enhancement, db_session):
    image_id = s3_image_for_enhancement["id"]
    token, user_email = create_user_and_get_token(client, db_session, "enh_ul_fail")
    auth_headers = {"Authorization": f"Bearer {token}"}
    enhancement_params = {"brightness_target": 1.1, "contrast_target": 1.1, "saturation_target": 1.1, "background_blur_radius": 0, "crop_rect": [0,0,10,10], "face_smooth_intensity": 0.0}
    request_body = {"image_id": image_id, "parameters": enhancement_params}
    with patch("main.storage_service.upload_file", side_effect=HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Simulated S3 Processed Upload Fail")) as mock_upload:
        response = client.post("/api/enhancement/apply", json=request_body, headers=auth_headers)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["original_image_id"] == image_id
    assert data["processed_image_id"] is None
    assert data["processed_image_path"] is None
    assert "S3 upload error: Simulated S3 Processed Upload Fail" in data["error"]
    mock_upload.assert_called_once()
    clear_user_from_db(SessionLocal(), user_email)

@pytest.fixture(scope="function")
def user_preset(client, db_session):
    token, user_email = create_user_and_get_token(client, db_session, "preset_owner")
    auth_headers = {"Authorization": f"Bearer {token}"}
    preset_params = {
        "brightness_target": 0.9, "contrast_target": 0.9, "saturation_target": 0.9,
        "background_blur_radius": 1, "crop_rect": [10,10,80,80], "face_smooth_intensity": 0.1
    }
    preset_create_data = {
        "preset_name": f"TestPreset_{uuid.uuid4().hex}",
        "parameters_json": json.dumps(preset_params)
    }
    create_preset_response = client.post("/api/users/presets/", json=preset_create_data, headers=auth_headers)
    assert create_preset_response.status_code == status.HTTP_201_CREATED, \
        f"Failed to create preset for testing: {create_preset_response.text}"
    preset_data = create_preset_response.json()
    return {"id": preset_data["id"], "user_token": token, "user_email": user_email, "auth_headers": auth_headers}

@mock_s3
def test_apply_preset_success(client, s3_image_for_enhancement, user_preset, db_session):
    image_id = s3_image_for_enhancement["id"]
    original_s3_key = s3_image_for_enhancement["s3_key"]
    preset_id = user_preset["id"]
    auth_headers = user_preset["auth_headers"]
    s3 = boto3.client("s3", region_name=os.environ["AWS_S3_REGION"])
    bucket_name = os.environ["AWS_S3_BUCKET_NAME"]
    s3.head_object(Bucket=bucket_name, Key=original_s3_key)
    request_body = {"image_id": image_id}
    response = client.post(f"/api/enhancement/apply-preset/{preset_id}", json=request_body, headers=auth_headers)
    assert response.status_code == status.HTTP_200_OK, response.text
    data = response.json()
    assert data["original_image_id"] == image_id
    assert data["processed_image_id"] is not None
    assert data["processed_image_path"] is not None
    assert "https://test-bucket.s3.us-east-1.amazonaws.com/processed_images/" in data["processed_image_path"]
    assert data["error"] is None
    db = SessionLocal()
    try:
        processed_img_record = db.query(models.Image).filter(models.Image.id == data["processed_image_id"]).first()
        assert processed_img_record is not None
        assert processed_img_record.filepath.startswith("processed_images/")
        assert "_preset_enhanced_" in processed_img_record.filepath
        s3.head_object(Bucket=bucket_name, Key=processed_img_record.filepath)
    finally:
        db.close()
    clear_user_from_db(SessionLocal(), user_preset["user_email"])

@mock_s3
def test_apply_preset_s3_original_download_failure(client, s3_image_for_enhancement, user_preset, db_session):
    image_id = s3_image_for_enhancement["id"]
    preset_id = user_preset["id"]
    auth_headers = user_preset["auth_headers"]
    request_body = {"image_id": image_id}
    with patch("main.storage_service.download_file", side_effect=HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Simulated S3 Original Not Found for Preset")) as mock_download:
        response = client.post(f"/api/enhancement/apply-preset/{preset_id}", json=request_body, headers=auth_headers)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "Could not retrieve original image" in data["error"] or "Simulated S3 Original Not Found for Preset" in data["error"]
    mock_download.assert_called_once()
    clear_user_from_db(SessionLocal(), user_preset["user_email"])

@mock_s3
def test_apply_preset_s3_processed_upload_failure(client, s3_image_for_enhancement, user_preset, db_session):
    image_id = s3_image_for_enhancement["id"]
    preset_id = user_preset["id"]
    auth_headers = user_preset["auth_headers"]
    request_body = {"image_id": image_id}
    with patch("main.storage_service.upload_file", side_effect=HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Simulated S3 Preset Upload Fail")) as mock_upload:
        response = client.post(f"/api/enhancement/apply-preset/{preset_id}", json=request_body, headers=auth_headers)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "S3 upload error: Simulated S3 Preset Upload Fail" in data["error"]
    mock_upload.assert_called_once()
    clear_user_from_db(SessionLocal(), user_preset["user_email"])

@mock_s3
def test_analysis_faces_success(client, s3_image_for_enhancement):
    image_id = s3_image_for_enhancement["id"]
    original_s3_key = s3_image_for_enhancement["s3_key"]
    s3 = boto3.client("s3", region_name=os.environ["AWS_S3_REGION"])
    bucket_name = os.environ["AWS_S3_BUCKET_NAME"]
    s3.head_object(Bucket=bucket_name, Key=original_s3_key)
    response = client.get(f"/api/analysis/faces/{image_id}")
    assert response.status_code == status.HTTP_200_OK, response.text
    data = response.json()
    assert data["image_id"] == image_id
    assert "faces" in data

@mock_s3
def test_analysis_faces_s3_download_failure(client, s3_image_for_enhancement):
    image_id = s3_image_for_enhancement["id"]
    with patch("main.storage_service.download_file", side_effect=HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Simulated S3 Download Fail for Faces")) as mock_download:
        response = client.get(f"/api/analysis/faces/{image_id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND, response.text
    data = response.json()
    assert "Simulated S3 Download Fail for Faces" in data["detail"]
    mock_download.assert_called_once()

@mock_s3
def test_analysis_quality_success(client, s3_image_for_enhancement):
    image_id = s3_image_for_enhancement["id"]
    response = client.get(f"/api/analysis/quality/{image_id}")
    assert response.status_code == status.HTTP_200_OK, response.text
    data = response.json()
    assert data["image_id"] == image_id
    assert "quality_metrics" in data

@mock_s3
def test_analysis_quality_s3_download_failure(client, s3_image_for_enhancement):
    image_id = s3_image_for_enhancement["id"]
    with patch("main.storage_service.download_file", side_effect=HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Simulated S3 Fail for Quality")) as mock_download:
        response = client.get(f"/api/analysis/quality/{image_id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND, response.text
    assert "Simulated S3 Fail for Quality" in response.json()["detail"]
    mock_download.assert_called_once()

@mock_s3
def test_enhancement_auto_success(client, s3_image_for_enhancement):
    image_id = s3_image_for_enhancement["id"]
    response = client.get(f"/api/enhancement/auto/{image_id}")
    assert response.status_code == status.HTTP_200_OK, response.text
    data = response.json()
    assert data["image_id"] == image_id
    assert "enhancement_parameters" in data

@mock_s3
def test_enhancement_auto_s3_download_failure(client, s3_image_for_enhancement):
    image_id = s3_image_for_enhancement["id"]
    with patch("main.storage_service.download_file", side_effect=HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Simulated S3 Fail for Auto Enhance")) as mock_download:
        response = client.get(f"/api/enhancement/auto/{image_id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND, response.text
    assert "Simulated S3 Fail for Auto Enhance" in response.json()["detail"]
    mock_download.assert_called_once()

@pytest.fixture
def mock_file_system_operations_fixture(mocker):
    mocker.patch("os.makedirs")
    mock_file_open = mock_open()
    mocker.patch("builtins.open", mock_file_open)
    mock_join = mocker.patch("os.path.join", return_value="mocked/path/to/file.jpg")
    return {
        "open": mock_file_open,
        "makedirs": os.makedirs,
        "join": mock_join
    }

@pytest.mark.usefixtures("mock_moderate_content_approve_fixture")
@mock_s3
class TestImageUploadMetadata:

    def test_upload_jpeg_metadata(self, client):
        s3 = boto3.client("s3", region_name=os.environ["AWS_S3_REGION"])
        s3.create_bucket(Bucket=os.environ["AWS_S3_BUCKET_NAME"])
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
        assert data["exif_orientation"] is None
        assert data["color_profile"] == "RGB"
        assert data["rejection_reason"] is None
        assert data["filepath"].startswith("original_images/")

    @mock_s3
    def test_upload_png_metadata(self, client):
        s3 = boto3.client("s3", region_name=os.environ["AWS_S3_REGION"])
        s3.create_bucket(Bucket=os.environ["AWS_S3_BUCKET_NAME"])
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
        assert data["exif_orientation"] is None
        assert data["color_profile"] == "RGB"
        assert data["rejection_reason"] is None
        assert data["filepath"].startswith("original_images/")

    @mock_s3
    def test_upload_jpeg_with_exif_orientation(self, client):
        s3 = boto3.client("s3", region_name=os.environ["AWS_S3_REGION"])
        s3.create_bucket(Bucket=os.environ["AWS_S3_BUCKET_NAME"])
        orientation_tag_id = 0x0112
        orientation_value = 3
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

    @mock_s3
    def test_upload_jpeg_no_exif_data(self, client):
        s3 = boto3.client("s3", region_name=os.environ["AWS_S3_REGION"])
        s3.create_bucket(Bucket=os.environ["AWS_S3_BUCKET_NAME"])
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

    @mock_s3
    def test_upload_jpeg_minimal_exif_no_orientation(self, client):
        s3 = boto3.client("s3", region_name=os.environ["AWS_S3_REGION"])
        s3.create_bucket(Bucket=os.environ["AWS_S3_BUCKET_NAME"])
        other_exif_tag_id = 0x010F
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
        assert data["exif_orientation"] is None
        assert data["color_profile"] == "RGB"
        assert data["rejection_reason"] is None

@pytest.mark.asyncio
async def test_moderate_content_general_exception_during_api_call():
    mock_image_pil = MagicMock()
    mock_image_pil.size = (1000, 800)
    mock_vision_client_instance = MagicMock()
    mock_vision_client_instance.face_detection.side_effect = Exception("Network timeout")
    with patch('main.PILImage.open', return_value=mock_image_pil) as mock_pil_open, \
         patch('main.vision.ImageAnnotatorClient', return_value=mock_vision_client_instance) as mock_vision_client_constructor:
        result = await moderate_image_content(MINIMAL_WEBP_BYTES_FOR_MODERATION)
        assert result.is_approved is False
        assert result.rejection_reason == "Content moderation check failed due to an internal error."
        mock_pil_open.assert_called_once()
        mock_vision_client_constructor.assert_called_once()
        mock_vision_client_instance.face_detection.assert_called_once()
        mock_vision_client_instance.safe_search_detection.assert_not_called()
