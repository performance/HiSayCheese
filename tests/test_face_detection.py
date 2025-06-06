import os
import shutil
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from PIL import Image, ImageDraw
import cv2 # For image processing and Haar cascade loading
import numpy as np # For image array manipulation if needed by cv2 directly
import uuid # For creating unique image IDs for DB records
from typing import List, Optional

# Ensure the main app and its modules can be imported
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app, UPLOAD_DIR # UPLOAD_DIR will be used for test images
from services.face_detection import detect_faces
from db import crud
from models import models
from db.database import SessionLocal, get_db, create_db_and_tables # For DB session and setup

# --- Test Setup & Fixtures ---

client = TestClient(app)

# Re-use the cleanup fixture from test_main.py by ensuring pytest discovers it.
# If test_main.py's cleanup_upload_dir_and_db_records is autouse=True, scope=function,
# it will apply to these tests as well. We will rely on this.
# If it's not in a conftest.py, we might need to explicitly import or define it here.
# For now, assuming it's discovered and active.

@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    # Ensures tables are created once per session, if not already by test_main.py's setup
    create_db_and_tables()

@pytest.fixture(scope="function")
def db_session() -> Session:
    """Provides a clean database session for each test function."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper function to create test images
def create_test_image_file(
    directory: str,
    filename: str,
    width: int,
    height: int,
    color: str = "white",
    faces_rects: Optional[List[List[int]]] = None # List of [x, y, w, h]
) -> str:
    """
    Creates an image file using PIL with optional 'faces' (rectangles).
    Saves it to directory/filename and returns the full path.
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    image = Image.new("RGB", (width, height), color=color)
    draw = ImageDraw.Draw(image)

    if faces_rects:
        for rect in faces_rects:
            x, y, w, h = rect
            # Draw a contrasting rectangle (e.g., black on white)
            face_color = "black" if color.lower() == "white" else "white"
            draw.rectangle([x, y, x + w, y + h], fill=face_color)

    filepath = os.path.join(directory, filename)
    image.save(filepath, "JPEG") # Save as JPEG for OpenCV compatibility
    return filepath

# Fixtures for test images
@pytest.fixture(scope="function")
def single_face_image_path():
    # UPLOAD_DIR is cleaned by test_main.py's fixture
    return create_test_image_file(
        UPLOAD_DIR, "single_face.jpg", 200, 200, "white", faces_rects=[[50, 50, 40, 40]]
    )

@pytest.fixture(scope="function")
def no_face_image_path():
    return create_test_image_file(
        UPLOAD_DIR, "no_face.jpg", 200, 200, "white", faces_rects=[]
    )

@pytest.fixture(scope="function")
def multi_face_image_path():
    return create_test_image_file(
        UPLOAD_DIR, "multi_face.jpg", 300, 300, "white",
        faces_rects=[[50, 50, 40, 40], [150, 150, 50, 50], [100, 200, 30, 30]]
    )

@pytest.fixture(scope="function")
def corrupted_image_path():
    filepath = os.path.join(UPLOAD_DIR, "corrupted.jpg")
    with open(filepath, "w") as f:
        f.write("This is not a valid image file.")
    return filepath

# --- Tests for detect_faces function (direct unit tests) ---

def test_detect_faces_single_face(single_face_image_path):
    """Test detection of a single face."""
    faces = detect_faces(single_face_image_path)
    assert len(faces) == 1
    face = faces[0]
    assert "box" in face
    assert isinstance(face["box"], list)
    assert len(face["box"]) == 4
    # Confidence for Haar is fixed at 1.0 in our implementation
    assert face["confidence"] == 1.0

def test_detect_faces_no_face(no_face_image_path):
    """Test with an image containing no faces."""
    faces = detect_faces(no_face_image_path)
    assert len(faces) == 0

def test_detect_faces_multiple_faces(multi_face_image_path):
    """Test detection of multiple faces."""
    # Note: Haar cascades can be sensitive. The number of detected faces
    # might not exactly match the number drawn if they are too close or sizes vary too much.
    # This test aims to see if multiple can be found, not necessarily an exact count.
    faces = detect_faces(multi_face_image_path)
    assert len(faces) > 1 # Expecting more than one, ideally 3
    # If test is flaky, might need to adjust rectangle sizes/positions or test expectation
    # For now, let's be optimistic about detecting all 3
    assert len(faces) == 3
    for face in faces:
        assert "box" in face
        assert isinstance(face["box"], list)
        assert len(face["box"]) == 4
        assert face["confidence"] == 1.0

def test_detect_faces_file_not_found():
    """Test with a non-existent image path."""
    with pytest.raises(FileNotFoundError):
        detect_faces("non_existent_image.jpg")

def test_detect_faces_corrupted_image(corrupted_image_path):
    """Test with a corrupted/invalid image file."""
    # Behavior depends on OpenCV's imread:
    # If imread returns None, detect_faces should raise ValueError.
    with pytest.raises(ValueError) as excinfo:
        detect_faces(corrupted_image_path)
    assert "Could not load image" in str(excinfo.value)

# --- Helper to create DB record for an image ---
def create_image_db_record(db: Session, filename: str, filepath: Optional[str],
                           width: int = 200, height: int = 200, format_type: str = "JPEG") -> models.Image:
    """
    Creates an image record in the database using crud.create_image.
    The ID is auto-generated by the database model (default=uuid.uuid4).
    """
    image_create_data = models.ImageCreate(
        filename=filename,
        filepath=filepath,
        filesize=1024,  # Dummy filesize
        mimetype='image/jpeg',  # Dummy mimetype
        width=width,
        height=height,
        format=format_type,
        # rejection_reason can be omitted if Optional and defaults to None
    )
    # crud.create_image also takes optional width, height, format, exif_orientation, color_profile
    # that override values in image_create_data if necessary, or complement them.
    # Here, we pass them explicitly for clarity matching typical usage in main.py
    db_image = crud.create_image(
        db=db,
        image=image_create_data, # This now contains width, height, format
        width=width, # Redundant if already in image_create_data, but crud.py might prioritize these args
        height=height,
        format=format_type
        # exif_orientation and color_profile can be omitted if not needed for the test
    )
    return db_image

# --- Tests for the API Endpoint /api/analysis/faces/{image_id} ---

def test_api_get_face_detections_single_face(db_session, single_face_image_path):
    db_image = create_image_db_record(db_session, filename="api_single.jpg", filepath=single_face_image_path)

    response = client.get(f"/api/analysis/faces/{db_image.id}")

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["image_id"] == str(db_image.id) # ID is UUID, converted to str in response
    assert len(data["faces"]) == 1
    face = data["faces"][0]
    assert "box" in face
    assert len(face["box"]) == 4
    assert face["confidence"] == 1.0

def test_api_get_face_detections_no_face(db_session, no_face_image_path):
    db_image = create_image_db_record(db_session, filename="api_no_face.jpg", filepath=no_face_image_path)

    response = client.get(f"/api/analysis/faces/{db_image.id}")

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["image_id"] == str(db_image.id)
    assert len(data["faces"]) == 0
    assert data["message"] == "No faces detected."

def test_api_get_face_detections_multiple_faces(db_session, multi_face_image_path):
    db_image = create_image_db_record(db_session, filename="api_multi_face.jpg", filepath=multi_face_image_path)

    response = client.get(f"/api/analysis/faces/{db_image.id}")

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["image_id"] == str(db_image.id)
    assert len(data["faces"]) == 3 # Expecting 3 based on fixture and direct test
    for face in data["faces"]:
        assert "box" in face
        assert len(face["box"]) == 4
        assert face["confidence"] == 1.0

def test_api_get_face_detections_image_id_not_found(db_session):
    non_existent_image_id = uuid.uuid4() # Generate a random UUID
    response = client.get(f"/api/analysis/faces/{non_existent_image_id}")

    assert response.status_code == 404, response.text
    assert f"Image with id {non_existent_image_id} not found" in response.json()["detail"]

def test_api_get_face_detections_image_record_no_filepath(db_session):
    # Create an image record with filepath=None
    db_image_no_fp = create_image_db_record(db_session, filename="no_filepath.jpg", filepath=None)

    response = client.get(f"/api/analysis/faces/{db_image_no_fp.id}")

    assert response.status_code == 404, response.text # As per main.py logic
    assert "Filepath for image id" in response.json()["detail"]
    assert "not available" in response.json()["detail"]

def test_api_get_face_detections_image_file_missing(db_session, single_face_image_path):
    # 1. Create DB record pointing to a valid file
    db_image = create_image_db_record(db_session, filename="initially_valid.jpg", filepath=single_face_image_path)

    # 2. Ensure file exists (it should from the fixture)
    assert os.path.exists(single_face_image_path)

    # 3. Delete the physical file
    os.remove(single_face_image_path)
    assert not os.path.exists(single_face_image_path)

    # 4. Call API
    response = client.get(f"/api/analysis/faces/{db_image.id}")

    # Expect 404 because detect_faces will raise FileNotFoundError
    assert response.status_code == 404, response.text
    assert "Image file not found for id" in response.json()["detail"]

# TODO: Consider a test for when cv2.data.haarcascades is missing, though this is more of an env issue.
# For now, assume 'haarcascade_frontalface_default.xml' is found by OpenCV.
# If detect_faces raises FileNotFoundError for the cascade file, the API should return 500.
# To test this, one might need to mock os.path.exists for the cascade file path.

# Example of how one might test for cascade file missing (requires more mocking setup):
# def test_api_detect_faces_cascade_file_missing(db_session, single_face_image_path, mocker):
#     db_image = create_image_db_record(db_session, "cascade_test.jpg", single_face_image_path)
#
#     # Mock os.path.exists specifically for the cascade file path check in detect_faces
#     # This is a bit advanced and depends on knowing the exact path used internally.
#     # Let's assume detect_faces constructs cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
#     # and then calls os.path.exists(cascade_path)
#     cascade_file_full_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
#     mocker.patch("os.path.exists", lambda path_arg: False if path_arg == cascade_file_full_path else os.path.exists(path_arg))
#
#     response = client.get(f"/api/analysis/faces/{db_image.id}")
#     assert response.status_code == 500 # Or whatever error main.py raises for this
#     assert "Haar Cascade model not found" in response.json()["detail"] # Or similar
# This test is commented out as it requires precise mocking and might be too fragile.
# The current FileNotFoundError from detect_faces for cascade file results in 500 in main.py
# (An unexpected error occurred during face detection.)
# To make it more specific, detect_faces would need to raise a custom exception or main.py needs to catch it better.
# For now, the existing FileNotFoundError for missing image file (handled as 404) is distinct.
# ValueError for corrupted image (handled as 500 "Invalid image file or format") is also distinct.
# A FileNotFoundError for the cascade XML currently falls into the generic Exception catch-all in the API endpoint, returning 500.

print(f"Test UPLOAD_DIR for face_detection tests: {os.path.abspath(UPLOAD_DIR)}")
print(f"CV2 Haar Cascades Path: {cv2.data.haarcascades}")
# Check if the default cascade file actually exists in the environment
default_cascade_file = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
if not os.path.exists(default_cascade_file):
    print(f"WARNING: Default Haar Cascade file NOT FOUND at {default_cascade_file}")
    print("Face detection tests will likely fail if they rely on the actual Haar cascade model.")
else:
    print(f"Default Haar Cascade file found at {default_cascade_file}")

# Simple check to ensure cv2 can load the cascade
try:
    face_cascade = cv2.CascadeClassifier(default_cascade_file)
    if face_cascade.empty():
        print("WARNING: cv2.CascadeClassifier loaded but is empty. Check OpenCV installation and cascade file.")
    else:
        print("cv2.CascadeClassifier loaded successfully with the default cascade.")
except Exception as e:
    print(f"WARNING: Error loading default Haar Cascade with cv2: {e}")
