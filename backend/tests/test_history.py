import pytest
import uuid
import json
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Generator

# Add project root to sys.path for module imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app
from db.database import SessionLocal, create_db_and_tables, get_db
from models import models
from db import crud
from auth_utils import hash_password, create_access_token

# --- Fixtures ---

@pytest.fixture(scope="session", autouse=True)
def setup_test_db_session_wide():
    """
    Creates all tables once per test session.
    Relies on function-scoped fixtures for data cleanup.
    """
    create_db_and_tables()
    # Create upload directories if they don't exist (needed by image creation/processing)
    os.makedirs("uploads/images/", exist_ok=True)
    os.makedirs("uploads/processed/", exist_ok=True)

@pytest.fixture(scope="function")
def db_session() -> Generator[Session, None, None]:
    """
    Provides a test database session that rolls back changes after each test.
    More robust for true isolation if dependency_overrides were used for get_db.
    For now, explicit cleanup is used.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        # Clean up database tables after each test to ensure independence
        # Order matters due to foreign key constraints
        db.query(models.EnhancementHistory).delete()
        db.query(models.Image).delete()
        db.query(models.User).delete()
        # Add other models if they are created and need cleanup
        db.commit()
        db.close()

@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)

# Helper to create a user directly in DB for testing prerequisites
def create_test_user_in_db(db: Session, email_prefix: str = "testuser_hist") -> models.User:
    email = f"{email_prefix}_{uuid.uuid4()}@example.com"
    password = "testpassword"
    hashed_pw = hash_password(password)
    user = models.User(email=email, hashed_password=hashed_pw)
    db.add(user)
    db.commit()
    db.refresh(user)
    # Store password for login if needed, not on the model itself
    setattr(user, '_test_password', password)
    return user

# Helper to create an image record directly in DB
def create_test_image_in_db(db: Session, filename_prefix: str = "test_image") -> models.Image:
    image_data = models.ImageCreate(
        filename=f"{filename_prefix}_{uuid.uuid4().hex}.jpg",
        filepath=f"/tmp/{filename_prefix}_{uuid.uuid4().hex}.jpg", # Dummy path
        filesize=12345,
        mimetype="image/jpeg",
        width=100,
        height=100,
        format="JPEG"
    )
    db_image = crud.create_image(db=db, image=image_data, width=100, height=100, format="JPEG")
    return db_image

# Fixture to create a user and return their details (id, email, password)
@pytest.fixture(scope="function")
def test_user(db_session: Session) -> models.User:
    return create_test_user_in_db(db_session)

# Fixture to get an auth token for a given user object
@pytest.fixture(scope="function")
def auth_token_for_user(client: TestClient):
    def _get_token(user: models.User) -> str:
        # Assumes user object has _test_password attribute set by create_test_user_in_db
        login_data = {"username": user.email, "password": getattr(user, '_test_password')}
        response = client.post("/api/auth/login", data=login_data)
        assert response.status_code == 200, f"Login failed for token generation: {response.text}"
        return response.json()["access_token"]
    return _get_token


# --- Unit Tests for CRUD operations ---

def test_create_enhancement_history(db_session: Session):
    user = create_test_user_in_db(db_session, "crud_user")
    original_image = create_test_image_in_db(db_session, "original")
    processed_image = create_test_image_in_db(db_session, "processed")

    history_data_in = models.EnhancementHistoryBase(
        original_image_id=original_image.id,
        processed_image_id=processed_image.id,
        parameters_json=json.dumps({"brightness": 0.5, "contrast": 0.2})
    )

    created_history = crud.create_enhancement_history(
        db=db_session,
        history_data=history_data_in,
        user_id=user.id
    )

    assert created_history is not None
    assert created_history.id is not None
    assert created_history.user_id == user.id
    assert created_history.original_image_id == original_image.id
    assert created_history.processed_image_id == processed_image.id
    assert json.loads(created_history.parameters_json) == {"brightness": 0.5, "contrast": 0.2}
    assert created_history.created_at is not None

    # Fetch from DB directly
    db_history_item = db_session.query(models.EnhancementHistory).filter(models.EnhancementHistory.id == created_history.id).first()
    assert db_history_item is not None
    assert db_history_item.user_id == user.id

def test_get_enhancement_history_by_user(db_session: Session):
    user1 = create_test_user_in_db(db_session, "user1_hist")
    user2 = create_test_user_in_db(db_session, "user2_hist")

    img1_orig = create_test_image_in_db(db_session, "u1_img1_orig")
    img1_proc = create_test_image_in_db(db_session, "u1_img1_proc")
    hist1_user1 = crud.create_enhancement_history(db_session, models.EnhancementHistoryBase(original_image_id=img1_orig.id, processed_image_id=img1_proc.id, parameters_json="{}"), user1.id)

    img2_orig = create_test_image_in_db(db_session, "u1_img2_orig")
    img2_proc = create_test_image_in_db(db_session, "u1_img2_proc")
    hist2_user1 = crud.create_enhancement_history(db_session, models.EnhancementHistoryBase(original_image_id=img2_orig.id, processed_image_id=img2_proc.id, parameters_json='{"param":1}'), user1.id)

    img3_orig = create_test_image_in_db(db_session, "u2_img3_orig")
    img3_proc = create_test_image_in_db(db_session, "u2_img3_proc")
    hist1_user2 = crud.create_enhancement_history(db_session, models.EnhancementHistoryBase(original_image_id=img3_orig.id, processed_image_id=img3_proc.id, parameters_json="{}"), user2.id)

    # Test for user1
    history_user1 = crud.get_enhancement_history_by_user(db_session, user_id=user1.id, skip=0, limit=10)
    assert len(history_user1) == 2
    assert {h.id for h in history_user1} == {hist1_user1.id, hist2_user1.id}
    # Check order (most recent first)
    assert history_user1[0].created_at >= history_user1[1].created_at
    if json.loads(history_user1[0].parameters_json) == {"param":1}: # hist2_user1 should be first if created later
        assert history_user1[0].id == hist2_user1.id
    else: # hist1_user1 was created later (less likely with sequential code but possible if clock moves)
        assert history_user1[0].id == hist1_user1.id


    # Test pagination for user1
    history_user1_limit1 = crud.get_enhancement_history_by_user(db_session, user_id=user1.id, skip=0, limit=1)
    assert len(history_user1_limit1) == 1
    assert history_user1_limit1[0].id == history_user1[0].id # Should be the most recent

    history_user1_skip1_limit1 = crud.get_enhancement_history_by_user(db_session, user_id=user1.id, skip=1, limit=1)
    assert len(history_user1_skip1_limit1) == 1
    assert history_user1_skip1_limit1[0].id == history_user1[1].id # Should be the second most recent

    # Test for user2
    history_user2 = crud.get_enhancement_history_by_user(db_session, user_id=user2.id, skip=0, limit=10)
    assert len(history_user2) == 1
    assert history_user2[0].id == hist1_user2.id

# --- Integration Tests for /api/enhancement/apply ---

MINIMAL_PNG_CONTENT_FOR_UPLOAD = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"

# Mock moderate_image_content to always approve for upload tests
@pytest.fixture(autouse=True) # Apply to all tests in this file/module
def mock_moderate_content_always_approve(mocker):
    mocker.patch(
        "main.moderate_image_content",
        return_value=models.ContentModerationResult(is_approved=True, rejection_reason=None)
    )

def test_apply_enhancement_creates_history(client: TestClient, db_session: Session, test_user: models.User, auth_token_for_user):
    token = auth_token_for_user(test_user)
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Upload an original image
    upload_response = client.post(
        "/api/images/upload",
        files={"file": ("original.png", MINIMAL_PNG_CONTENT_FOR_UPLOAD, "image/png")},
        headers=headers # Some endpoints might require auth for upload too, assume not for now or add if needed
    )
    assert upload_response.status_code == 201, upload_response.text
    original_image_data = upload_response.json()
    original_image_id = original_image_data["id"]

    # 2. Prepare enhancement parameters
    enhancement_params = {
        "brightness_target": 1.1,
        "contrast_target": 1.1,
        "saturation_target": 1.1,
        "background_blur_radius": 0,
        "crop_rect": [0, 0, 100, 100], # Assuming original image is at least 100x100
        "face_smooth_intensity": 0.0
    }
    request_payload = {
        "image_id": original_image_id,
        "parameters": enhancement_params
    }

    # 3. Call /api/enhancement/apply
    apply_response = client.post("/api/enhancement/apply", json=request_payload, headers=headers)
    assert apply_response.status_code == 200, apply_response.text
    apply_data = apply_response.json()

    assert apply_data["original_image_id"] == original_image_id
    assert apply_data["processed_image_id"] is not None
    assert apply_data["processed_image_path"] is not None

    # 4. Verify DB records
    # Check new Image record for processed image
    processed_image_db = db_session.query(models.Image).filter(models.Image.id == apply_data["processed_image_id"]).first()
    assert processed_image_db is not None
    assert processed_image_db.id == uuid.UUID(apply_data["processed_image_id"])
    assert "enhanced" in processed_image_db.filename # Check naming convention

    # Check EnhancementHistory record
    history_record = db_session.query(models.EnhancementHistory).filter(
        models.EnhancementHistory.user_id == test_user.id,
        models.EnhancementHistory.original_image_id == original_image_id,
        models.EnhancementHistory.processed_image_id == processed_image_db.id
    ).first()

    assert history_record is not None
    assert history_record.user_id == test_user.id
    assert json.loads(history_record.parameters_json) == enhancement_params

# --- Integration Tests for /api/users/history ---

def test_get_user_history_authenticated(client: TestClient, db_session: Session, test_user: models.User, auth_token_for_user):
    token = auth_token_for_user(test_user)
    headers = {"Authorization": f"Bearer {token}"}

    # Create some history: Upload image, then apply enhancement
    upload_resp = client.post("/api/images/upload", files={"file": ("hist_orig.png", MINIMAL_PNG_CONTENT_FOR_UPLOAD, "image/png")}, headers=headers)
    orig_img_id = upload_resp.json()["id"]
    apply_payload = {"image_id": orig_img_id, "parameters": {"brightness_target": 1.0, "contrast_target": 1.0, "saturation_target": 1.0, "background_blur_radius": 0, "crop_rect": [0,0,1,1], "face_smooth_intensity": 0}}
    client.post("/api/enhancement/apply", json=apply_payload, headers=headers) # Creates one history item

    # Get history
    history_response = client.get("/api/users/history", headers=headers)
    assert history_response.status_code == 200, history_response.text
    history_data = history_response.json()

    assert isinstance(history_data, list)
    assert len(history_data) == 1
    assert history_data[0]["original_image_id"] == orig_img_id
    assert history_data[0]["user_id"] == str(test_user.id) # Ensure user_id is in response and matches
    assert "parameters_json" in history_data[0]

def test_get_user_history_unauthenticated(client: TestClient):
    response = client.get("/api/users/history")
    assert response.status_code == 401 # HTTP_401_UNAUTHORIZED

def test_get_user_history_pagination(client: TestClient, db_session: Session, test_user: models.User, auth_token_for_user):
    token = auth_token_for_user(test_user)
    headers = {"Authorization": f"Bearer {token}"}

    # Create 3 history items
    for i in range(3):
        upload_resp = client.post("/api/images/upload", files={"file": (f"page_img_{i}.png", MINIMAL_PNG_CONTENT_FOR_UPLOAD, "image/png")}, headers=headers)
        orig_img_id = upload_resp.json()["id"]
        apply_payload = {"image_id": orig_img_id, "parameters": {"brightness_target": 1.0+i*0.1, "contrast_target": 1.0, "saturation_target": 1.0, "background_blur_radius": 0, "crop_rect": [0,0,1,1], "face_smooth_intensity": 0}}
        client.post("/api/enhancement/apply", json=apply_payload, headers=headers)

    # Test limit
    response_limit2 = client.get("/api/users/history?limit=2", headers=headers)
    assert response_limit2.status_code == 200
    assert len(response_limit2.json()) == 2

    # Test skip and limit
    response_skip1_limit1 = client.get("/api/users/history?skip=1&limit=1", headers=headers)
    assert response_skip1_limit1.status_code == 200
    data_skip1_limit1 = response_skip1_limit1.json()
    assert len(data_skip1_limit1) == 1

    # Verify items are different and in order (most recent first)
    response_all = client.get("/api/users/history?limit=3", headers=headers)
    all_items = response_all.json()
    assert data_skip1_limit1[0]["id"] == all_items[1]["id"] # Second item from all

def test_user_cannot_see_other_user_history(client: TestClient, db_session: Session, test_user: models.User, auth_token_for_user):
    user_a = test_user # Re-use fixture for user A
    token_a = auth_token_for_user(user_a)
    headers_a = {"Authorization": f"Bearer {token_a}"}

    # Create history for User A
    upload_resp_a = client.post("/api/images/upload", files={"file": ("user_a_img.png", MINIMAL_PNG_CONTENT_FOR_UPLOAD, "image/png")}, headers=headers_a)
    orig_img_id_a = upload_resp_a.json()["id"]
    apply_payload_a = {"image_id": orig_img_id_a, "parameters": {"brightness_target": 1.1, "contrast_target": 1.0, "saturation_target": 1.0, "background_blur_radius": 0, "crop_rect": [0,0,1,1], "face_smooth_intensity": 0}}
    client.post("/api/enhancement/apply", json=apply_payload_a, headers=headers_a)


    # Create User B and their history
    user_b_details = create_test_user_in_db(db_session, "userB_other") # Using direct db creation for simplicity
    token_b = auth_token_for_user(user_b_details) # Get token for user B
    headers_b = {"Authorization": f"Bearer {token_b}"}

    upload_resp_b = client.post("/api/images/upload", files={"file": ("user_b_img.png", MINIMAL_PNG_CONTENT_FOR_UPLOAD, "image/png")}, headers=headers_b)
    orig_img_id_b = upload_resp_b.json()["id"]
    apply_payload_b = {"image_id": orig_img_id_b, "parameters": {"brightness_target": 1.2, "contrast_target": 1.0, "saturation_target": 1.0, "background_blur_radius": 0, "crop_rect": [0,0,1,1], "face_smooth_intensity": 0}}
    client.post("/api/enhancement/apply", json=apply_payload_b, headers=headers_b)

    # Log in as User A and fetch history
    history_response_a = client.get("/api/users/history", headers=headers_a)
    assert history_response_a.status_code == 200
    history_data_a = history_response_a.json()

    assert len(history_data_a) == 1 # Should only see their own history
    assert history_data_a[0]["original_image_id"] == orig_img_id_a
    assert history_data_a[0]["user_id"] == str(user_a.id)

    # Log in as User B and fetch history (optional, but good for sanity check)
    history_response_b = client.get("/api/users/history", headers=headers_b)
    assert history_response_b.status_code == 200
    history_data_b = history_response_b.json()
    assert len(history_data_b) == 1
    assert history_data_b[0]["original_image_id"] == orig_img_id_b
    assert history_data_b[0]["user_id"] == str(user_b_details.id)

# Placeholder for a test that ensures processed_image_id can be null in history
# This would require a way to trigger enhancement failure or metadata-only change via endpoint
# or directly creating such a record and testing its retrieval.
# For now, crud.create_enhancement_history allows processed_image_id to be None
# in EnhancementHistoryBase, so the CRUD part is covered.
# def test_history_with_null_processed_image_id(...):
# pass
