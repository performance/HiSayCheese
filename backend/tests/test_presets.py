import pytest
import uuid
import json
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Generator

# Add project root to sys.path for module imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app # Main FastAPI app
from db.database import SessionLocal, create_db_and_tables, get_db
from models import models # All SQLAlchemy models and Pydantic schemas
from db import crud
from auth_utils import hash_password, create_access_token

# --- Fixtures ---

@pytest.fixture(scope="session", autouse=True)
def setup_test_db_session_wide():
    create_db_and_tables()
    os.makedirs("uploads/images/", exist_ok=True)
    os.makedirs("uploads/processed/", exist_ok=True)

@pytest.fixture(scope="function")
def db_session() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        # Cleanup order: Presets, History, Images, Users
        db.query(models.UserPreset).delete()
        db.query(models.EnhancementHistory).delete()
        db.query(models.Image).delete()
        db.query(models.User).delete()
        db.commit()
        db.close()

@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)

# Helper to create a user directly in DB for testing prerequisites
def create_test_user_in_db(db: Session, email_prefix: str = "testuser_preset") -> models.User:
    email = f"{email_prefix}_{uuid.uuid4()}@example.com"
    password = "testpassword"
    hashed_pw = hash_password(password)
    user = models.User(email=email, hashed_password=hashed_pw)
    db.add(user)
    db.commit()
    db.refresh(user)
    setattr(user, '_test_password', password) # Store password for login helper
    return user

# Fixture to create a user and return their model instance
@pytest.fixture(scope="function")
def test_user(db_session: Session) -> models.User:
    return create_test_user_in_db(db_session)

# Fixture to get an auth token for a given user object
@pytest.fixture(scope="function")
def auth_token_for_user(client: TestClient):
    def _get_token(user: models.User) -> str:
        login_data = {"username": user.email, "password": getattr(user, '_test_password')}
        response = client.post("/api/auth/login", data=login_data)
        assert response.status_code == 200, f"Login failed for token generation: {response.text}"
        return response.json()["access_token"]
    return _get_token

# Helper to create an image record directly in DB
def create_test_image_in_db(db: Session, filename_prefix: str = "preset_test_img") -> models.Image:
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

# Fixture for creating a test image (needed for apply-preset tests)
@pytest.fixture
def test_image(db_session: Session) -> models.Image:
    return create_test_image_in_db(db_session)

# Mock moderate_image_content to always approve for upload tests in apply-preset
@pytest.fixture(autouse=True)
def mock_moderate_content_always_approve(mocker):
    mocker.patch(
        "main.moderate_image_content",
        return_value=models.ContentModerationResult(is_approved=True, rejection_reason=None)
    )

# --- Unit Tests for Preset CRUD operations ---

DEFAULT_PRESET_PARAMS = json.dumps({"brightness_target": 1.2, "contrast_target": 1.1, "saturation_target": 1.0, "background_blur_radius": 0, "crop_rect": [0,0,100,100], "face_smooth_intensity": 0.1})

def test_create_user_preset(db_session: Session, test_user: models.User):
    preset_data_in = models.PresetCreate(
        preset_name="My Test Preset",
        parameters_json=DEFAULT_PRESET_PARAMS
    )
    created_preset = crud.create_user_preset(db_session, preset_data_in, test_user.id)

    assert created_preset is not None
    assert created_preset.id is not None
    assert created_preset.user_id == test_user.id
    assert created_preset.preset_name == "My Test Preset"
    assert created_preset.parameters_json == DEFAULT_PRESET_PARAMS
    assert created_preset.created_at is not None

    db_preset = db_session.query(models.UserPreset).filter(models.UserPreset.id == created_preset.id).first()
    assert db_preset is not None
    assert db_preset.preset_name == "My Test Preset"

def test_get_user_presets_by_user(db_session: Session):
    user_a = create_test_user_in_db(db_session, "userA_getpresets")
    user_b = create_test_user_in_db(db_session, "userB_getpresets")

    crud.create_user_preset(db_session, models.PresetCreate(preset_name="A_Preset1", parameters_json="{}"), user_a.id)
    crud.create_user_preset(db_session, models.PresetCreate(preset_name="A_Preset2", parameters_json="{}"), user_a.id)
    crud.create_user_preset(db_session, models.PresetCreate(preset_name="B_Preset1", parameters_json="{}"), user_b.id)

    presets_a = crud.get_user_presets_by_user(db_session, user_a.id)
    assert len(presets_a) == 2
    assert {p.preset_name for p in presets_a} == {"A_Preset1", "A_Preset2"}
    assert presets_a[0].created_at >= presets_a[1].created_at # Check order

    presets_b = crud.get_user_presets_by_user(db_session, user_b.id)
    assert len(presets_b) == 1
    assert presets_b[0].preset_name == "B_Preset1"

def test_get_user_preset(db_session: Session, test_user: models.User):
    preset_in_data = models.PresetCreate(preset_name="Specific Preset", parameters_json="{}")
    created_preset = crud.create_user_preset(db_session, preset_in_data, test_user.id)

    # Correct user
    fetched_preset = crud.get_user_preset(db_session, created_preset.id, test_user.id)
    assert fetched_preset is not None
    assert fetched_preset.id == created_preset.id

    # Incorrect user
    other_user = create_test_user_in_db(db_session, "other_user_get")
    fetched_preset_other_user = crud.get_user_preset(db_session, created_preset.id, other_user.id)
    assert fetched_preset_other_user is None

    # Non-existent preset
    fetched_non_existent = crud.get_user_preset(db_session, uuid.uuid4(), test_user.id)
    assert fetched_non_existent is None

def test_update_user_preset(db_session: Session, test_user: models.User):
    preset_in_data = models.PresetCreate(preset_name="Original Name", parameters_json=DEFAULT_PRESET_PARAMS)
    db_preset = crud.create_user_preset(db_session, preset_in_data, test_user.id)

    update_data = models.PresetUpdate(preset_name="Updated Name")
    updated_preset = crud.update_user_preset(db_session, db_preset.id, update_data, test_user.id)
    assert updated_preset is not None
    assert updated_preset.preset_name == "Updated Name"
    assert updated_preset.parameters_json == DEFAULT_PRESET_PARAMS # Unchanged

    update_data_params = models.PresetUpdate(parameters_json=json.dumps({"brightness": 0.8}))
    updated_preset_params = crud.update_user_preset(db_session, db_preset.id, update_data_params, test_user.id)
    assert updated_preset_params is not None
    assert updated_preset_params.preset_name == "Updated Name" # From previous update
    assert json.loads(updated_preset_params.parameters_json) == {"brightness": 0.8}

    # Attempt update by wrong user
    other_user = create_test_user_in_db(db_session, "other_user_update")
    update_fail_data = models.PresetUpdate(preset_name="Should Not Update")
    failed_update = crud.update_user_preset(db_session, db_preset.id, update_fail_data, other_user.id)
    assert failed_update is None

    db_session.refresh(db_preset) # Re-fetch from DB
    assert db_preset.preset_name == "Updated Name" # Ensure it wasn't changed by other_user

def test_delete_user_preset(db_session: Session, test_user: models.User):
    preset_in_data = models.PresetCreate(preset_name="To Be Deleted", parameters_json="{}")
    db_preset = crud.create_user_preset(db_session, preset_in_data, test_user.id)
    preset_id = db_preset.id

    # Attempt delete by wrong user
    other_user = create_test_user_in_db(db_session, "other_user_delete")
    deleted_by_other = crud.delete_user_preset(db_session, preset_id, other_user.id)
    assert deleted_by_other is None
    assert db_session.query(models.UserPreset).filter(models.UserPreset.id == preset_id).first() is not None # Still exists

    # Delete by correct user
    deleted_preset = crud.delete_user_preset(db_session, preset_id, test_user.id)
    assert deleted_preset is not None
    assert deleted_preset.id == preset_id
    assert db_session.query(models.UserPreset).filter(models.UserPreset.id == preset_id).first() is None # Deleted


# --- Integration Tests for Preset CRUD API Endpoints ---

MINIMAL_PNG_CONTENT_FOR_UPLOAD = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"

def test_create_preset_api(client: TestClient, test_user: models.User, auth_token_for_user):
    token = auth_token_for_user(test_user)
    headers = {"Authorization": f"Bearer {token}"}
    preset_payload = {"preset_name": "API Preset", "parameters_json": DEFAULT_PRESET_PARAMS}

    response = client.post("/api/users/presets", json=preset_payload, headers=headers)
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["preset_name"] == "API Preset"
    assert json.loads(data["parameters_json"]) == json.loads(DEFAULT_PRESET_PARAMS)
    assert data["user_id"] == str(test_user.id)

    # Test unauthenticated
    response_unauth = client.post("/api/users/presets", json=preset_payload)
    assert response_unauth.status_code == status.HTTP_401_UNAUTHORIZED

def test_list_presets_api(client: TestClient, db_session: Session, test_user: models.User, auth_token_for_user):
    user_a = test_user
    token_a = auth_token_for_user(user_a)
    headers_a = {"Authorization": f"Bearer {token_a}"}

    crud.create_user_preset(db_session, models.PresetCreate(preset_name="UserA_P1", parameters_json="{}"), user_a.id)
    crud.create_user_preset(db_session, models.PresetCreate(preset_name="UserA_P2", parameters_json="{}"), user_a.id)

    user_b = create_test_user_in_db(db_session, "userB_list_api")
    crud.create_user_preset(db_session, models.PresetCreate(preset_name="UserB_P1", parameters_json="{}"), user_b.id)

    response = client.get("/api/users/presets", headers=headers_a)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data) == 2
    assert {p["preset_name"] for p in data} == {"UserA_P1", "UserA_P2"}

    # Test unauthenticated
    response_unauth = client.get("/api/users/presets")
    assert response_unauth.status_code == status.HTTP_401_UNAUTHORIZED

def test_get_specific_preset_api(client: TestClient, db_session: Session, test_user: models.User, auth_token_for_user):
    token = auth_token_for_user(test_user)
    headers = {"Authorization": f"Bearer {token}"}

    preset = crud.create_user_preset(db_session, models.PresetCreate(preset_name="Specific", parameters_json="{}"), test_user.id)

    # Get own preset
    response = client.get(f"/api/users/presets/{preset.id}", headers=headers)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["id"] == str(preset.id)

    # Non-existent
    response_404 = client.get(f"/api/users/presets/{uuid.uuid4()}", headers=headers)
    assert response_404.status_code == status.HTTP_404_NOT_FOUND

    # Preset owned by another user
    other_user = create_test_user_in_db(db_session, "other_specific_api")
    other_preset = crud.create_user_preset(db_session, models.PresetCreate(preset_name="OtherUserPreset", parameters_json="{}"), other_user.id)
    response_other_404 = client.get(f"/api/users/presets/{other_preset.id}", headers=headers) # Using User A's token
    assert response_other_404.status_code == status.HTTP_404_NOT_FOUND

    # Unauthenticated
    response_unauth = client.get(f"/api/users/presets/{preset.id}")
    assert response_unauth.status_code == status.HTTP_401_UNAUTHORIZED

def test_update_preset_api(client: TestClient, db_session: Session, test_user: models.User, auth_token_for_user):
    token = auth_token_for_user(test_user)
    headers = {"Authorization": f"Bearer {token}"}
    preset = crud.create_user_preset(db_session, models.PresetCreate(preset_name="UpdateMe", parameters_json="{}"), test_user.id)

    update_payload = {"preset_name": "Updated via API"}
    response = client.put(f"/api/users/presets/{preset.id}", json=update_payload, headers=headers)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["preset_name"] == "Updated via API"
    assert data["id"] == str(preset.id)

    # Partial update (only params)
    update_payload_params = {"parameters_json": DEFAULT_PRESET_PARAMS}
    response_params = client.put(f"/api/users/presets/{preset.id}", json=update_payload_params, headers=headers)
    assert response_params.status_code == status.HTTP_200_OK
    assert json.loads(response_params.json()["parameters_json"]) == json.loads(DEFAULT_PRESET_PARAMS)
    assert response_params.json()["preset_name"] == "Updated via API" # Name from previous update

    # Try to update non-existent
    response_404 = client.put(f"/api/users/presets/{uuid.uuid4()}", json=update_payload, headers=headers)
    assert response_404.status_code == status.HTTP_404_NOT_FOUND

    # Unauthenticated
    response_unauth = client.put(f"/api/users/presets/{preset.id}", json=update_payload)
    assert response_unauth.status_code == status.HTTP_401_UNAUTHORIZED

def test_delete_preset_api(client: TestClient, db_session: Session, test_user: models.User, auth_token_for_user):
    token = auth_token_for_user(test_user)
    headers = {"Authorization": f"Bearer {token}"}
    preset = crud.create_user_preset(db_session, models.PresetCreate(preset_name="DeleteMe", parameters_json="{}"), test_user.id)

    response = client.delete(f"/api/users/presets/{preset.id}", headers=headers)
    assert response.status_code == status.HTTP_204_NO_CONTENT

    # Verify it's gone
    response_get = client.get(f"/api/users/presets/{preset.id}", headers=headers)
    assert response_get.status_code == status.HTTP_404_NOT_FOUND

    # Try to delete non-existent
    response_404 = client.delete(f"/api/users/presets/{uuid.uuid4()}", headers=headers)
    assert response_404.status_code == status.HTTP_404_NOT_FOUND

    # Unauthenticated
    response_unauth = client.delete(f"/api/users/presets/{preset.id}") # Preset ID might be invalid now
    assert response_unauth.status_code == status.HTTP_401_UNAUTHORIZED


# --- Integration Tests for Apply Preset API Endpoint ---

def test_apply_preset_api_success(client: TestClient, db_session: Session, test_user: models.User, auth_token_for_user, test_image: models.Image):
    token = auth_token_for_user(test_user)
    headers = {"Authorization": f"Bearer {token}"}

    preset_params_dict = {"brightness_target": 1.5, "contrast_target": 0.9, "saturation_target": 1.2, "background_blur_radius": 5, "crop_rect": [10,10,80,80], "face_smooth_intensity": 0.3}
    preset = crud.create_user_preset(db_session, models.PresetCreate(preset_name="ApplyMePreset", parameters_json=json.dumps(preset_params_dict)), test_user.id)

    apply_payload = {"image_id": str(test_image.id)}
    response = client.post(f"/api/enhancement/apply-preset/{preset.id}", json=apply_payload, headers=headers)

    assert response.status_code == status.HTTP_200_OK, response.text
    data = response.json()
    assert data["original_image_id"] == str(test_image.id)
    assert data["processed_image_id"] is not None
    assert data["processed_image_path"] is not None # Actual path check might be too brittle

    # Verify DB records
    processed_img_id = uuid.UUID(data["processed_image_id"])
    processed_img_db = db_session.query(models.Image).filter(models.Image.id == processed_img_id).first()
    assert processed_img_db is not None

    history_record = db_session.query(models.EnhancementHistory).filter(
        models.EnhancementHistory.user_id == test_user.id,
        models.EnhancementHistory.original_image_id == test_image.id,
        models.EnhancementHistory.processed_image_id == processed_img_id
    ).first()
    assert history_record is not None

    history_params = json.loads(history_record.parameters_json)
    assert history_params["applied_preset_id"] == str(preset.id)
    # Check if other params from preset are there (excluding the added applied_preset_id)
    for k, v in preset_params_dict.items():
        assert history_params[k] == v


def test_apply_preset_api_unauthenticated(client: TestClient, test_user: models.User, test_image: models.Image, db_session: Session): # Added db_session
    preset = crud.create_user_preset(db_session, models.PresetCreate(preset_name="UnauthPreset", parameters_json=DEFAULT_PRESET_PARAMS), test_user.id)
    apply_payload = {"image_id": str(test_image.id)}
    response = client.post(f"/api/enhancement/apply-preset/{preset.id}", json=apply_payload)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_apply_preset_api_preset_not_found(client: TestClient, test_user: models.User, auth_token_for_user, test_image: models.Image):
    token = auth_token_for_user(test_user)
    headers = {"Authorization": f"Bearer {token}"}
    apply_payload = {"image_id": str(test_image.id)}
    response = client.post(f"/api/enhancement/apply-preset/{uuid.uuid4()}", json=apply_payload, headers=headers)
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "Preset not found" in response.json()["detail"]

def test_apply_preset_api_preset_not_owned(client: TestClient, db_session: Session, test_user: models.User, auth_token_for_user, test_image: models.Image):
    user_a_token = auth_token_for_user(test_user) # User A is our main test_user
    headers_a = {"Authorization": f"Bearer {user_a_token}"}

    user_b = create_test_user_in_db(db_session, "userB_own_preset")
    preset_b = crud.create_user_preset(db_session, models.PresetCreate(preset_name="UserBPreset", parameters_json=DEFAULT_PRESET_PARAMS), user_b.id)

    apply_payload = {"image_id": str(test_image.id)} # Image can be any existing image for this test
    response = client.post(f"/api/enhancement/apply-preset/{preset_b.id}", json=apply_payload, headers=headers_a) # User A tries to use User B's preset
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "Preset not found or not owned by user" in response.json()["detail"]


def test_apply_preset_api_image_not_found(client: TestClient, test_user: models.User, auth_token_for_user, db_session: Session): # Added db_session
    token = auth_token_for_user(test_user)
    headers = {"Authorization": f"Bearer {token}"}
    preset = crud.create_user_preset(db_session, models.PresetCreate(preset_name="ImageNotFoundPreset", parameters_json=DEFAULT_PRESET_PARAMS), test_user.id)

    apply_payload = {"image_id": str(uuid.uuid4())} # Non-existent image ID
    response = client.post(f"/api/enhancement/apply-preset/{preset.id}", json=apply_payload, headers=headers)
    # This endpoint returns ProcessedImageResponse with an error message, not 404 directly for image issues
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["error"] is not None
    assert f"Image with id {apply_payload['image_id']} not found" in data["error"]


def test_apply_preset_api_invalid_parameters_in_preset(client: TestClient, test_user: models.User, auth_token_for_user, db_session: Session, test_image: models.Image):
    token = auth_token_for_user(test_user)
    headers = {"Authorization": f"Bearer {token}"}

    # Create a preset with parameters that won't match EnhancementRequestParams (e.g., wrong type or missing field)
    invalid_params_json = json.dumps({"brightness_target": "not_a_float", "contrast_target": 1.0}) # Missing other required fields
    preset_with_invalid_params = crud.create_user_preset(db_session,
        models.PresetCreate(preset_name="InvalidParamsPreset", parameters_json=invalid_params_json),
        test_user.id
    )

    apply_payload = {"image_id": str(test_image.id)}
    response = client.post(f"/api/enhancement/apply-preset/{preset_with_invalid_params.id}", json=apply_payload, headers=headers)

    # The endpoint should catch Pydantic validation error when parsing preset_parameters_dict
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Preset parameters are invalid" in response.json()["detail"]

# Minimal PNG content for upload if needed for apply-preset that involves creating a new image via upload first
# MINIMAL_PNG_CONTENT_FOR_UPLOAD = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
# Helper to upload an image and return its ID (if apply-preset test needs fresh image upload)
# def upload_test_image_get_id(client: TestClient, token: str, filename: str = "uploaded_for_preset.png") -> str:
#     response = client.post(
#         "/api/images/upload",
#         files={"file": (filename, MINIMAL_PNG_CONTENT_FOR_UPLOAD, "image/png")},
#         headers={"Authorization": f"Bearer {token}"}
#     )
#     assert response.status_code == 201, response.text
#     return response.json()["id"]
# @pytest.fixture
# def uploaded_image_id(client: TestClient, auth_token_for_user, test_user):
#    token = auth_token_for_user(test_user)
#    return upload_test_image_get_id(client, token)
