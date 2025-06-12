import pytest
import io
import os
import uuid
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

try:
    from backend.main import app
    from backend.services.storage_service import _current_test_storage_service_instance_holder, StorageService
    from backend.auth_utils import get_current_user, oauth2_scheme
    from backend.models.models import User as UserModel
except ImportError as e:
    raise ImportError(f"Failed to import backend components: {e}. Check PYTHONPATH or module paths.")

client = TestClient(app)

TEST_TOKEN_USER_ID = "00000000-0000-0000-0000-000000000000"

print(f"DEBUG TestUploads Module: id of _current_test_storage_service_instance_holder: {id(_current_test_storage_service_instance_holder)}")

@pytest.fixture
def mock_s3_init_and_methods_env_var():
    # print(f"DEBUG Fixture setup: id of holder at fixture start: {id(_current_test_storage_service_instance_holder)}")
    original_env_value = os.environ.get("TEST_MODE_NO_S3_CONNECT")
    os.environ["TEST_MODE_NO_S3_CONNECT"] = "true"
    _current_test_storage_service_instance_holder['instance'] = None
    yield
    if original_env_value is None:
        if "TEST_MODE_NO_S3_CONNECT" in os.environ:
            del os.environ["TEST_MODE_NO_S3_CONNECT"]
    else:
        os.environ["TEST_MODE_NO_S3_CONNECT"] = original_env_value
    _current_test_storage_service_instance_holder['instance'] = None

@pytest.fixture
def mock_s3_upload_error_env_var():
    original_env_value = os.environ.get("TEST_MODE_S3_UPLOAD_ERROR")
    os.environ["TEST_MODE_S3_UPLOAD_ERROR"] = "true"
    yield
    if original_env_value is None:
        if "TEST_MODE_S3_UPLOAD_ERROR" in os.environ:
            del os.environ["TEST_MODE_S3_UPLOAD_ERROR"]
    else:
        os.environ["TEST_MODE_S3_UPLOAD_ERROR"] = original_env_value

def get_test_storage_service_instance() -> StorageService:
    print(f"DEBUG get_test_storage_service_instance: id of holder: {id(_current_test_storage_service_instance_holder)}")
    print(f"DEBUG get_test_storage_service_instance: holder content: {_current_test_storage_service_instance_holder}")
    instance = _current_test_storage_service_instance_holder.get('instance')
    if instance is None:
        raise Exception("StorageService test instance not found in holder. "
                        "Ensure TEST_MODE_NO_S3_CONNECT is active, the endpoint "
                        "depending on StorageService has been called, and that the holder is not cleared prematurely.")
    return instance

def test_upload_image_success(mock_s3_init_and_methods_env_var):
    headers = {"Authorization": "Bearer test-only-token"}
    file_content = b"fake image data for upload"
    filename = "test_image.jpg"
    files = {"file": (filename, io.BytesIO(file_content), "image/jpeg")}

    response = client.post("/api/users/upload-image", files=files, headers=headers)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"

    json_response = response.json()
    storage_service_mocked_instance = get_test_storage_service_instance()

    returned_object_key = json_response["object_key"]
    assert returned_object_key.startswith(f"uploads/{TEST_TOKEN_USER_ID}/")
    assert returned_object_key.endswith(f".{filename.split('.')[-1]}")

    assert json_response["file_url"] == f"http://mocked_s3_url/{returned_object_key}"

    storage_service_mocked_instance.upload_file.assert_called_once()
    args_upload, kwargs_upload = storage_service_mocked_instance.upload_file.call_args
    assert kwargs_upload["object_key"] == returned_object_key
    assert kwargs_upload["content_type"] == "image/jpeg"

    storage_service_mocked_instance.get_public_url.assert_called_once_with(returned_object_key)


def test_upload_image_storage_error(mock_s3_init_and_methods_env_var, mock_s3_upload_error_env_var):
    headers = {"Authorization": "Bearer test-only-token"}
    file_content = b"image data for storage error test"
    files = {"file": ("error_image.png", io.BytesIO(file_content), "image/png")}

    response = client.post("/api/users/upload-image", files=files, headers=headers)

    assert response.status_code == 500, f"Expected 500, got {response.status_code}. Response: {response.text}"
    json_response = response.json()
    assert "Simulated S3 Error from __init__ based on TEST_MODE_S3_UPLOAD_ERROR" in str(json_response["detail"])

    storage_service_mocked_instance = get_test_storage_service_instance()
    storage_service_mocked_instance.upload_file.assert_called_once()


def test_upload_image_unauthenticated():
    original_oauth_override = app.dependency_overrides.pop(oauth2_scheme, None)
    original_user_override = app.dependency_overrides.pop(get_current_user, None)
    original_storage_override = app.dependency_overrides.pop(StorageService, None)

    file_content = b"unauthenticated image data"
    files = {"file": ("unauth_test.jpg", io.BytesIO(file_content), "image/jpeg")}
    response = client.post("/api/users/upload-image", files=files) # No headers
    assert response.status_code == 401

    if original_oauth_override: app.dependency_overrides[oauth2_scheme] = original_oauth_override
    if original_user_override: app.dependency_overrides[get_current_user] = original_user_override
    if original_storage_override: app.dependency_overrides[StorageService] = original_storage_override
