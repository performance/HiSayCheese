import pytest
import io
import os
import uuid
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

try:
    from backend.main import app
    from backend.services.storage_service import StorageService
    from backend.auth_utils import get_current_user, oauth2_scheme
    from backend.models.models import User as UserModel
except ImportError as e:
    raise ImportError(f"Failed to import backend components: {e}. Check PYTHONPATH or module paths.")

client = TestClient(app)

TEST_TOKEN_USER_ID = "00000000-0000-0000-0000-000000000000"

@pytest.fixture
def mock_s3_init_env_var():
    original_env_value = os.environ.get("TEST_MODE_NO_S3_CONNECT")
    os.environ["TEST_MODE_NO_S3_CONNECT"] = "true"
    yield
    if original_env_value is None:
        if "TEST_MODE_NO_S3_CONNECT" in os.environ:
            del os.environ["TEST_MODE_NO_S3_CONNECT"]
    else:
        os.environ["TEST_MODE_NO_S3_CONNECT"] = original_env_value

@pytest.fixture
def configured_mock_storage_service_instance(): # Renamed to be explicit about instance
    mock_instance = MagicMock(spec=StorageService)

    def dynamic_get_public_url(object_key):
        # print(f"DEBUG: Mock get_public_url called with {object_key}") # Optional debug
        return f"http://mocked_s3_url/{object_key}"
    mock_instance.get_public_url.side_effect = dynamic_get_public_url

    # For upload_file, the return value itself is not part of the endpoint's response object_key.
    # The endpoint generates its own object_key.
    mock_instance.upload_file.return_value = "placeholder_upload_path"
    return mock_instance

# Reverting to app.dependency_overrides for StorageService
def test_upload_image_success(configured_mock_storage_service_instance, mock_s3_init_env_var):
    # mock_s3_init_env_var ensures StorageService.__init__ is benign.
    # Now, override StorageService with our pre-configured mock instance.
    original_storage_override = app.dependency_overrides.get(StorageService)
    app.dependency_overrides[StorageService] = lambda: configured_mock_storage_service_instance

    headers = {"Authorization": "Bearer test-only-token"}
    file_content = b"fake image data for upload"
    files = {"file": ("test_image.jpg", io.BytesIO(file_content), "image/jpeg")}

    response = client.post("/api/users/upload-image", files=files, headers=headers)

    # Cleanup StorageService override immediately
    if original_storage_override: app.dependency_overrides[StorageService] = original_storage_override
    elif StorageService in app.dependency_overrides: del app.dependency_overrides[StorageService]

    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    json_response = response.json()

    returned_object_key = json_response["object_key"]
    assert returned_object_key.startswith(f"uploads/{TEST_TOKEN_USER_ID}/")
    assert returned_object_key.endswith(".jpg")

    assert json_response["file_url"] == f"http://mocked_s3_url/{returned_object_key}"

    configured_mock_storage_service_instance.upload_file.assert_called_once()
    _args_upload, kwargs_upload = configured_mock_storage_service_instance.upload_file.call_args
    assert kwargs_upload["object_key"] == returned_object_key
    assert kwargs_upload["content_type"] == "image/jpeg"

    configured_mock_storage_service_instance.get_public_url.assert_called_once_with(returned_object_key)


def test_upload_image_storage_error(configured_mock_storage_service_instance, mock_s3_init_env_var):
    # mock_s3_init_env_var ensures StorageService.__init__ is benign.
    original_storage_override = app.dependency_overrides.get(StorageService)
    app.dependency_overrides[StorageService] = lambda: configured_mock_storage_service_instance

    simulated_error_message = "Simulated S3 Network Error by Test"
    configured_mock_storage_service_instance.upload_file.side_effect = Exception(simulated_error_message)
    # Clear any mock calls from previous tests if the instance is somehow reused (pytest fixtures usually prevent this)
    configured_mock_storage_service_instance.upload_file.reset_mock()


    headers = {"Authorization": "Bearer test-only-token"}
    file_content = b"image data for storage error test"
    files = {"file": ("error_image.png", io.BytesIO(file_content), "image/png")}

    response = client.post("/api/users/upload-image", files=files, headers=headers)

    # Cleanup StorageService override immediately
    if original_storage_override: app.dependency_overrides[StorageService] = original_storage_override
    elif StorageService in app.dependency_overrides: del app.dependency_overrides[StorageService]

    assert response.status_code == 500, f"Expected 500, got {response.status_code}. Response: {response.text}"
    json_response = response.json()
    assert simulated_error_message in str(json_response["detail"])
    configured_mock_storage_service_instance.upload_file.assert_called_once()


def test_upload_image_unauthenticated():
    original_oauth_override = app.dependency_overrides.pop(oauth2_scheme, None)
    original_user_override = app.dependency_overrides.pop(get_current_user, None)
    original_storage_override = app.dependency_overrides.pop(StorageService, None)

    file_content = b"unauthenticated image data"
    files = {"file": ("unauth_test.jpg", io.BytesIO(file_content), "image/jpeg")}
    response = client.post("/api/users/upload-image", files=files)
    assert response.status_code == 401

    if original_oauth_override: app.dependency_overrides[oauth2_scheme] = original_oauth_override
    if original_user_override: app.dependency_overrides[get_current_user] = original_user_override
    if original_storage_override: app.dependency_overrides[StorageService] = original_storage_override
