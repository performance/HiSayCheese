import os
import uuid
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
import time # For performance testing

# Add the project root to the Python path to allow direct import of main and services
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app # Assuming your FastAPI app instance is named 'app' in main.py
from services.image_quality import analyze_image_quality
# For setting up test DB and creating image records:
from db.database import get_db, create_db_and_tables, SessionLocal
from db import crud
from models.models import ImageCreate # Make sure this matches your actual model name

# --- Test Setup and Fixtures ---
TEST_UPLOAD_DIR = "test_uploads_quality_temp/" # Unique name to avoid conflicts

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    os.makedirs(TEST_UPLOAD_DIR, exist_ok=True)
    # Ensure this uses a test database or is safe to run repeatedly
    create_db_and_tables()
    yield
    # Teardown: Remove created files and directories
    for item in os.listdir(TEST_UPLOAD_DIR):
        try:
            item_path = os.path.join(TEST_UPLOAD_DIR, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
        except Exception as e:
            print(f"Error removing test file: {item_path} - {e}") # Or use logger
    try:
        if os.path.exists(TEST_UPLOAD_DIR) and not os.listdir(TEST_UPLOAD_DIR): # Only remove if empty
            os.rmdir(TEST_UPLOAD_DIR)
        elif os.path.exists(TEST_UPLOAD_DIR):
            print(f"Test directory {TEST_UPLOAD_DIR} not empty, not removing.")
    except Exception as e:
        print(f"Error removing test directory: {TEST_UPLOAD_DIR} - {e}")


@pytest.fixture(scope="function")
def client():
    return TestClient(app)

@pytest.fixture(scope="function")
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_dummy_image(filepath: str, color: tuple, width: int = 100, height: int = 100):
    image = np.full((height, width, 3), color, dtype=np.uint8)
    cv2.imwrite(filepath, image)
    return filepath

def create_gradient_image(filepath: str, width: int = 256, height: int = 100, gray_range=(0, 255)):
    image = np.zeros((height, width), dtype=np.uint8)
    min_val, max_val = gray_range
    for i in range(width):
        image[:, i] = min_val + int((max_val - min_val) * (i / width))
    # Convert grayscale to BGR before saving if needed by other functions,
    # though analyze_image_quality converts to grayscale anyway
    bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(filepath, bgr_image)
    return filepath

# --- Test Cases for the Service ---

def test_analyze_quality_dark_image():
    image_path = create_dummy_image(os.path.join(TEST_UPLOAD_DIR, "dark.png"), (10, 10, 10))
    start_time = time.time()
    result = analyze_image_quality(image_path)
    end_time = time.time()
    assert result["brightness"] < 0.3, f"Brightness {result['brightness']}"
    assert "image is dark" in result["insights"]
    assert (end_time - start_time) < 2.0, f"Analysis took {end_time - start_time:.2f}s"
    if os.path.exists(image_path): os.remove(image_path)

def test_analyze_quality_bright_image():
    image_path = create_dummy_image(os.path.join(TEST_UPLOAD_DIR, "bright.png"), (245, 245, 245))
    result = analyze_image_quality(image_path)
    assert result["brightness"] > 0.7, f"Brightness {result['brightness']}"
    assert "image is bright" in result["insights"]
    if os.path.exists(image_path): os.remove(image_path)

def test_analyze_quality_low_contrast_image():
    image_path = create_dummy_image(os.path.join(TEST_UPLOAD_DIR, "low_contrast.png"), (128, 128, 128))
    result = analyze_image_quality(image_path)
    assert 0.49 < result["brightness"] < 0.51, f"Brightness {result['brightness']}" # Uniform gray (128) is 0.5 brightness
    assert result["contrast"] < 20, f"Contrast {result['contrast']}"
    assert "image has low contrast" in result["insights"]
    assert "image has balanced brightness" in result["insights"]
    if os.path.exists(image_path): os.remove(image_path)

def test_analyze_quality_high_contrast_image():
    # Full black to white gradient
    image_path_gradient = create_gradient_image(os.path.join(TEST_UPLOAD_DIR, "high_contrast_gradient.png"), gray_range=(0,255))
    result_gradient = analyze_image_quality(image_path_gradient)
    assert 0.49 < result_gradient["brightness"] < 0.51, f"Brightness {result_gradient['brightness']} for gradient"
    assert result_gradient["contrast"] > 60, f"Contrast {result_gradient['contrast']} for gradient"
    assert "image has high contrast" in result_gradient["insights"]
    if os.path.exists(image_path_gradient): os.remove(image_path_gradient)

    # Half black, half white image
    bw_image_data = np.zeros((100, 100, 3), dtype=np.uint8)
    bw_image_data[0:50, :] = 0   # Top half black
    bw_image_data[50:100, :] = 255 # Bottom half white
    bw_path = os.path.join(TEST_UPLOAD_DIR, "bw_contrast.png")
    cv2.imwrite(bw_path, bw_image_data)
    result_bw = analyze_image_quality(bw_path)
    assert 0.49 < result_bw["brightness"] < 0.51, f"Brightness {result_bw['brightness']} for B&W"
    assert result_bw["contrast"] > 60, f"Contrast {result_bw['contrast']} for B&W"
    assert "image has high contrast" in result_bw["insights"]
    if os.path.exists(bw_path): os.remove(bw_path)


def test_analyze_quality_balanced_image():
    image_path = create_gradient_image(os.path.join(TEST_UPLOAD_DIR, "balanced.png"), gray_range=(64, 191))
    result = analyze_image_quality(image_path)
    assert 0.3 <= result["brightness"] <= 0.7, f"Brightness {result['brightness']}"
    assert 20 <= result["contrast"] <= 60, f"Contrast {result['contrast']}"
    assert "image has balanced brightness" in result["insights"]
    assert "image has balanced contrast" in result["insights"]
    if os.path.exists(image_path): os.remove(image_path)

def test_analyze_quality_file_not_found():
    with pytest.raises(FileNotFoundError):
        analyze_image_quality("non_existent_image.png")

def test_analyze_invalid_image_file():
    invalid_image_path = os.path.join(TEST_UPLOAD_DIR, "invalid.png")
    with open(invalid_image_path, "w") as f:
        f.write("this is not an image")
    with pytest.raises(ValueError): # analyze_image_quality raises ValueError for unreadable images
        analyze_image_quality(invalid_image_path)
    if os.path.exists(invalid_image_path): os.remove(invalid_image_path)

# --- Test Cases for the API Endpoint ---

def create_mock_image_in_db(db_session, filename="test.png", content_type="image/png", color=(128,128,128), width=100, height=100, is_gradient=False, gray_range=(0,255)):
    dummy_filepath = os.path.join(TEST_UPLOAD_DIR, filename)
    if is_gradient:
        create_gradient_image(dummy_filepath, width=width, height=height, gray_range=gray_range)
    else:
        create_dummy_image(dummy_filepath, color, width=width, height=height)

    image_data = ImageCreate(
        filename=filename,
        filepath=dummy_filepath,
        filesize=os.path.getsize(dummy_filepath),
        mimetype=content_type,
        width=width,
        height=height,
        format="PNG",
        rejection_reason=None,
        exif_orientation=None,
        color_profile=None
    )
    db_image = crud.create_image(
        db=db_session,
        image=image_data,
        width=image_data.width,
        height=image_data.height,
        format=image_data.format,
        exif_orientation=image_data.exif_orientation,
        color_profile=image_data.color_profile
    )
    return db_image, dummy_filepath


def test_api_get_image_quality_success(client, db_session):
    db_image, dummy_filepath = create_mock_image_in_db(db_session, filename="api_test_quality.png", color=(128,128,128)) # Uniform gray

    response = client.get(f"/api/analysis/quality/{db_image.id}")

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["image_id"] == str(db_image.id)
    assert "quality_metrics" in data
    assert 0.49 < data["quality_metrics"]["brightness"] < 0.51 # Brightness of 128/255
    assert data["quality_metrics"]["contrast"] < 20 # Low contrast for uniform gray
    assert "image has low contrast" in data["insights"]
    assert "image has balanced brightness" in data["insights"]

    if os.path.exists(dummy_filepath): os.remove(dummy_filepath)


def test_api_get_image_quality_not_found(client):
    non_existent_uuid = uuid.uuid4()
    response = client.get(f"/api/analysis/quality/{non_existent_uuid}")
    assert response.status_code == 404
    assert response.json()["detail"] == f"Image with id {non_existent_uuid} not found."

def test_api_get_image_quality_filepath_missing(client, db_session):
    image_data_no_path = ImageCreate(
        filename="no_path.png", filepath=None, filesize=100, mimetype="image/png",
        width=100,height=100,format="PNG", exif_orientation=None, color_profile=None, rejection_reason=None
    )
    db_image_no_path = crud.create_image(
        db=db_session, image=image_data_no_path, width=100,height=100,format="PNG",
        exif_orientation=None, color_profile=None
    )

    response = client.get(f"/api/analysis/quality/{db_image_no_path.id}")
    assert response.status_code == 404
    assert "Filepath for image id" in response.json()["detail"]


def test_api_get_image_quality_actual_file_deleted(client, db_session):
    db_image, dummy_filepath = create_mock_image_in_db(db_session, filename="deleted_file_test.png")

    if os.path.exists(dummy_filepath): os.remove(dummy_filepath)

    response = client.get(f"/api/analysis/quality/{db_image.id}")
    assert response.status_code == 404
    assert f"Image file not found at path {dummy_filepath}" in response.json()["detail"]

def test_api_get_image_quality_dark_image(client, db_session):
    db_image, dummy_filepath = create_mock_image_in_db(db_session, filename="api_dark_test.png", color=(10,10,10))

    response = client.get(f"/api/analysis/quality/{db_image.id}")

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["quality_metrics"]["brightness"] < 0.3
    assert "image is dark" in data["insights"]

    if os.path.exists(dummy_filepath): os.remove(dummy_filepath)

def test_api_get_image_quality_bright_image(client, db_session):
    db_image, dummy_filepath = create_mock_image_in_db(db_session, filename="api_bright_test.png", color=(245,245,245))

    response = client.get(f"/api/analysis/quality/{db_image.id}")

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["quality_metrics"]["brightness"] > 0.7
    assert "image is bright" in data["insights"]

    if os.path.exists(dummy_filepath): os.remove(dummy_filepath)

def test_api_get_image_quality_high_contrast(client, db_session):
    db_image, dummy_filepath = create_mock_image_in_db(
        db_session,
        filename="api_high_contrast_gradient.png",
        is_gradient=True,
        gray_range=(0,255),
        width=256, # Match gradient default width
        height=100 # Match gradient default height
    )

    response = client.get(f"/api/analysis/quality/{db_image.id}")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["quality_metrics"]["contrast"] > 60
    assert "image has high contrast" in data["insights"]
    assert 0.49 < data["quality_metrics"]["brightness"] < 0.51 # Gradient 0-255 has brightness ~0.5

    if os.path.exists(dummy_filepath): os.remove(dummy_filepath)
