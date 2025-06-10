import pytest
import io
import os
import cv2
import numpy as np
import tempfile # Though not directly used if we mock services that take paths
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

try:
    from backend.main import app
    from backend.services.storage_service import _current_test_storage_service_instance_holder, StorageService
    # For type hinting or direct patching if needed, though most will be via string path
    # from backend.services.face_detection import detect_faces
    # from backend.services.image_quality import analyze_image_quality
    # from backend.services.apply_image_modifications import apply_all_enhancements
except ImportError as e:
    raise ImportError(f"Failed to import backend components: {e}. Check PYTHONPATH or module paths.")

client = TestClient(app)

TEST_USER_ID = "00000000-0000-0000-0000-000000000000" # From test-only-token

@pytest.fixture
def test_env_setup_for_images_router():
    original_env_value_no_s3 = os.environ.get("TEST_MODE_NO_S3_CONNECT")
    original_env_value_s3_err = os.environ.get("TEST_MODE_S3_UPLOAD_ERROR")

    os.environ["TEST_MODE_NO_S3_CONNECT"] = "true"
    _current_test_storage_service_instance_holder['instance'] = None # Clear at start

    yield

    if original_env_value_no_s3 is None:
        if "TEST_MODE_NO_S3_CONNECT" in os.environ:
             del os.environ["TEST_MODE_NO_S3_CONNECT"]
    else:
        os.environ["TEST_MODE_NO_S3_CONNECT"] = original_env_value

    if original_env_value_s3_err is None:
        if "TEST_MODE_S3_UPLOAD_ERROR" in os.environ:
            del os.environ["TEST_MODE_S3_UPLOAD_ERROR"]
    else:
        os.environ["TEST_MODE_S3_UPLOAD_ERROR"] = original_env_value

    _current_test_storage_service_instance_holder['instance'] = None # Clear at end

def get_mocked_storage_service() -> StorageService: # Type hint with actual class for clarity
    instance = _current_test_storage_service_instance_holder.get('instance')
    if instance is None:
        # This might happen if the endpoint call itself fails before StorageService is instantiated by DI
        # Or if the fixture didn't run, or the holder logic in StorageService has an issue.
        # For tests where an endpoint call is made, this should ideally find the instance.
        # If a test *doesn't* call an endpoint that uses StorageService, but still calls this, it will fail.
        # One way to "prime" it for configuration is to make a dummy call or have a dedicated setup step.
        # However, the current design relies on StorageService.__init__ creating the method mocks.
        # We configure these mocks *after* instance creation (after endpoint call) for assertions,
        # or use @patch for more direct control over methods for specific behaviors like raising errors.
        raise Exception("StorageService test instance not found in holder. Ensure TEST_MODE_NO_S3_CONNECT is active and an endpoint calling StorageService has been hit OR StorageService was instantiated in test setup.")
    return instance # Instance methods are already MagicMocks due to __init__ modification

# --- Helper to create dummy image bytes ---
def create_dummy_image_bytes(fmt=".jpg", color=(128,128,128)) -> bytes:
    img = np.full((10, 10, 3), color, dtype=np.uint8)
    is_success, buffer = cv2.imencode(fmt, img)
    if not is_success:
        raise RuntimeError(f"Could not encode dummy image to {fmt}")
    return buffer.tobytes()

# --- Tests for POST /api/images/analyze ---

@patch('backend.routers.images.analyze_image_quality')
@patch('backend.routers.images.detect_faces')
def test_analyze_image_success(mock_detect_faces, mock_analyze_quality, test_env_setup_for_images_router):
    mock_detect_faces.return_value = {"faces": [{"box": [10,10,50,50], "confidence": 0.99}]}
    mock_analyze_quality.return_value = {"brightness": 0.5, "contrast": 50.0, "insights": ["good"]}

    mock_object_key = "test/dummy_image_for_analyze.jpg"

    # The StorageService instance will be created by Depends(StorageService)
    # Its download_file method is already a MagicMock due to __init__ modification.
    # We don't need to configure its side_effect if the patched functions don't use the file content.
    # If they expected a valid file path, download_file would need to handle that.
    # For this test, detect_faces and analyze_image_quality are fully mocked,
    # so they won't interact with the file system path passed by the endpoint.

    response = client.post(
        "/api/images/analyze",
        json={"object_key": mock_object_key},
        headers={"Authorization": "Bearer test-only-token"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["object_key"] == mock_object_key
    assert data["face_detection"] == {"faces": [{"box": [10,10,50,50], "confidence": 0.99}]}
    assert data["image_quality"]["brightness"] == 0.5

    mock_detect_faces.assert_called_once()
    mock_analyze_quality.assert_called_once()

    # Check that download_file on the StorageService instance was called
    storage_mock_instance = get_mocked_storage_service()
    storage_mock_instance.download_file.assert_called_once()
    # We can also assert the args if needed, e.g., that object_key was correct
    args, _kwargs = storage_mock_instance.download_file.call_args
    assert args[0] == mock_object_key # object_key is the first positional arg

def test_analyze_image_s3_download_fails(test_env_setup_for_images_router):
    mock_object_key = "test/nonexistent.jpg"

    # We need to configure the download_file mock on the instance *before* the endpoint calls it.
    # This is tricky. Instead, we patch it on the class for this test.
    expected_exception = HTTPException(status_code=404, detail="File not found in S3 mock")
    with patch.object(StorageService, 'download_file', side_effect=expected_exception) as mock_download:
        response = client.post(
            "/api/images/analyze",
            json={"object_key": mock_object_key},
            headers={"Authorization": "Bearer test-only-token"}
        )
        mock_download.assert_called_once_with(object_key=mock_object_key, destination_path=mock_download.call_args[1]['destination_path'])

    assert response.status_code == 404
    assert "File not found in S3 mock" in response.json()["detail"]

@patch('backend.routers.images.analyze_image_quality') # Keep this one to prevent its execution
@patch('backend.routers.images.detect_faces', side_effect=ValueError("Bad image for face detection"))
def test_analyze_image_face_detection_error(mock_detect_faces_err, mock_analyze_quality_noop, test_env_setup_for_images_router):
    # mock_analyze_quality_noop will prevent analyze_image_quality from running if detect_faces fails first
    mock_object_key = "test/bad_image_for_faces.jpg"

    response = client.post(
        "/api/images/analyze",
        json={"object_key": mock_object_key},
        headers={"Authorization": "Bearer test-only-token"}
    )

    assert response.status_code == 400 # As per endpoint's ValueError handling
    assert "Image format error or unprocessable image: Bad image for face detection" in response.json()["detail"]
    mock_detect_faces_err.assert_called_once()
    mock_analyze_quality_noop.assert_not_called() # Should not be called if detect_faces fails

# --- Tests for POST /api/images/apply-enhancements ---

@patch('backend.routers.images.detect_faces') # If face smoothing > 0, this will be called
@patch('backend.routers.images.apply_all_enhancements')
def test_apply_enhancements_success(mock_apply_all, mock_detect_faces_for_smooth, test_env_setup_for_images_router):
    dummy_processed_image = create_dummy_image_bytes(".jpg", color=(100,100,100))
    # apply_all_enhancements returns a numpy array
    mock_apply_all.return_value = cv2.imdecode(np.frombuffer(dummy_processed_image, np.uint8), cv2.IMREAD_COLOR)

    mock_detect_faces_for_smooth.return_value = {"faces": [{"box": [10,10,50,50]}]}


    request_data = {
        "original_object_key": "test/original.jpg",
        "enhancements": {
            "brightness_target": 0.1, # Mapped to 10 in backend service if (val-0.5)*X, X=100 -> (0.1-0.5)*100 = -40
                                      # If 0 is no change, this is +0.1 for service if service expects 0-1 range.
                                      # The service apply_brightness_contrast expects brightness as offset (beta).
                                      # Pydantic model default is 0.0.
            "contrast_target": 1.2,   # Pydantic model default is 1.0
            "face_smooth_intensity": 0.5 # This will trigger detect_faces call
        }
    }

    response = client.post(
        "/api/images/apply-enhancements",
        json=request_data,
        headers={"Authorization": "Bearer test-only-token"}
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["original_object_key"] == request_data["original_object_key"]
    assert data["enhanced_object_key"].startswith(f"enhanced_images/{TEST_USER_ID}/")
    assert data["enhanced_object_key"].endswith(".jpg") # Original was .jpg
    assert data["enhanced_file_url"].startswith(f"http://mocked_s3_url/enhanced_images/{TEST_USER_ID}/")

    storage_mock_instance = get_mocked_storage_service()
    storage_mock_instance.download_file.assert_called_once()
    storage_mock_instance.upload_file.assert_called_once()
    mock_apply_all.assert_called_once()
    mock_detect_faces_for_smooth.assert_called_once() # Because face_smooth_intensity > 0

    # Check args of apply_all_enhancements
    args_apply_all, _ = mock_apply_all.call_args
    assert "enhancement_params" in args_apply_all[1]
    assert args_apply_all[1]["enhancement_params"]["brightness_target"] == 0.1
    assert args_apply_all[1]["enhancement_params"]["face_boxes"] == [[10,10,50,50]]


def test_apply_enhancements_download_fails(test_env_setup_for_images_router):
    expected_exception = HTTPException(status_code=404, detail="S3 Original Not Found")
    with patch.object(StorageService, 'download_file', side_effect=expected_exception) as mock_download:
        response = client.post(
            "/api/images/apply-enhancements",
            json={
                "original_object_key": "test/nonexistent.jpg",
                "enhancements": {"brightness_target": 0.0} # Minimal valid enhancements
            },
            headers={"Authorization": "Bearer test-only-token"}
        )
        mock_download.assert_called_once()
    assert response.status_code == 404
    assert "S3 Original Not Found" in response.json()["detail"]


@patch('backend.routers.images.apply_all_enhancements') # To prevent it from running
def test_apply_enhancements_upload_fails(mock_apply_all_noop, test_env_setup_for_images_router):
    mock_apply_all_noop.return_value = create_dummy_image(10,10) # Needs to return an image

    expected_exception = Exception("S3 Upload Mock Error")
    # Patch the upload_file method on the StorageService class for this test's scope
    with patch.object(StorageService, 'upload_file', side_effect=expected_exception) as mock_upload:
        response = client.post(
            "/api/images/apply-enhancements",
            json={
                "original_object_key": "test/original_for_upload_fail.jpg",
                "enhancements": {}
            },
            headers={"Authorization": "Bearer test-only-token"}
        )
        mock_upload.assert_called_once()

    assert response.status_code == 500
    assert "S3 Upload Mock Error" in response.json()["detail"]


@patch('backend.routers.images.apply_all_enhancements', return_value=None) # Simulate processing returning None
def test_apply_enhancements_image_processing_returns_none(mock_apply_all_none, test_env_setup_for_images_router):
    response = client.post(
        "/api/images/apply-enhancements",
        json={
            "original_object_key": "test/original_for_proc_fail.jpg",
            "enhancements": {}
        },
        headers={"Authorization": "Bearer test-only-token"}
    )
    assert response.status_code == 400 # As per endpoint logic for modified_image_np is None
    assert "Image processing failed" in response.json()["detail"]
    mock_apply_all_none.assert_called_once()

@patch('backend.routers.images.apply_all_enhancements', side_effect=ValueError("Bad image for enhancement"))
def test_apply_enhancements_image_processing_raises_valueerror(mock_apply_all_value_err, test_env_setup_for_images_router):
    response = client.post(
        "/api/images/apply-enhancements",
        json={
            "original_object_key": "test/original_for_proc_value_err.jpg",
            "enhancements": {}
        },
        headers={"Authorization": "Bearer test-only-token"}
    )
    assert response.status_code == 400 # As per endpoint logic for ValueError
    assert "Image format error or unprocessable image (apply-enhancements): Bad image for enhancement" in response.json()["detail"]
    mock_apply_all_value_err.assert_called_once()

# test_upload_image_unauthenticated can be added from previous versions if needed
# It's already in the current test_uploads.py in the prompt.
# For brevity here, I'll assume it would be copied over or is already there.
# Adding it for completeness of this file.
def test_apply_enhancements_unauthenticated():
    # Ensure no auth overrides are active for this test
    original_oauth_override = app.dependency_overrides.pop(oauth2_scheme, None)
    original_user_override = app.dependency_overrides.pop(get_current_user, None)
    original_storage_override = app.dependency_overrides.pop(StorageService, None)

    response = client.post(
        "/api/images/apply-enhancements",
        json={
            "original_object_key": "test/original.jpg",
            "enhancements": {}
        }
        # No Authorization header
    )
    assert response.status_code == 401

    if original_oauth_override: app.dependency_overrides[oauth2_scheme] = original_oauth_override
    if original_user_override: app.dependency_overrides[get_current_user] = original_user_override
    if original_storage_override: app.dependency_overrides[StorageService] = original_storage_override
