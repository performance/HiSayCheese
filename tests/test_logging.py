import pytest
import json
import uuid
import logging
from fastapi.testclient import TestClient
from main import app
from datetime import timedelta
from jose import jwt, JWTError
from config import SECRET_KEY, ALGORITHM

client = TestClient(app)

def is_valid_uuid(uuid_to_test, version=4):
    try:
        uuid_obj = uuid.UUID(uuid_to_test, version=version)
    except ValueError:
        return False
    return str(uuid_obj) == uuid_to_test

@pytest.fixture(autouse=True)
def clear_log_captures(caplog):
    caplog.clear()

@pytest.fixture(scope="module")
def test_user_token():
    unique_email = f"logtestuser_{uuid.uuid4().hex}@example.com"
    user_data = {
        "email": unique_email,
        "password": "ValidPassword123!",
        "full_name": "Log Test User"
    }
    print(f"Registering user with payload: {user_data}")
    reg_response = client.post("/api/auth/register", json=user_data, headers={"Content-Type": "application/json"})
    print(f"Registration response status: {reg_response.status_code}")
    print(f"Registration response content: {reg_response.text}")
    if reg_response.status_code != 201:
         pytest.fail(f"Failed to register test user {user_data['email']} for token generation. Status: {reg_response.status_code}, Response: {reg_response.text}")

    login_data = {"username": user_data["email"], "password": user_data["password"]}
    token_response = client.post("/api/auth/login", data=login_data)
    if token_response.status_code != 200:
        pytest.fail(f"Failed to get token for test user {user_data['email']}: Status {token_response.status_code}, Body {token_response.text}")

    token_json = token_response.json()

    try:
        payload = jwt.decode(token_json['access_token'], SECRET_KEY, algorithms=[ALGORITHM])
        user_id_from_token = payload.get("user_id")
        if not user_id_from_token:
             user_id_from_token = payload.get("sub")
             if not user_id_from_token:
                 pytest.fail("user_id (or sub as fallback) not found in JWT token payload.")
    except jwt.JWTError as e:
        pytest.fail(f"JWT decoding error: {e}")

    return {
        "access_token": token_json["access_token"],
        "user_id": str(user_id_from_token)
    }

def test_structured_json_logging_health_endpoint(caplog):
    caplog.set_level(logging.INFO)
    response = client.get("/health")
    assert response.status_code in [200, 404]

    request_id_from_header = response.headers.get("X-Request-ID")
    assert request_id_from_header is not None, "X-Request-ID header missing from response."

    found_record_attributes = False
    for record in caplog.records:
        if record.name == 'main' and "Processing /health endpoint." in record.message:
            assert hasattr(record, 'request_id'), "LogRecord missing request_id attribute"
            assert record.request_id is not None, "request_id attribute is None"
            assert len(record.request_id) == 32, f"request_id '{record.request_id}' has unexpected length"
            assert "X-Request-ID" in response.headers, "X-Request-ID header missing from response"
            assert response.headers["X-Request-ID"] == record.request_id, "X-Request-ID header does not match record.request_id"
            assert record.levelname == "INFO"
            assert record.name == "main"
            found_record_attributes = True
            break
    assert found_record_attributes, "Target LogRecord from /health with correct attributes not found."

def test_error_logging_and_sentry_integration(caplog):
    caplog.set_level(logging.ERROR)
    response = client.get("/test-error")
    assert response.status_code == 500

    request_id_from_header = response.headers.get("X-Request-ID")
    assert request_id_from_header is not None

    found_record_attributes = False
    for record in caplog.records:
        if record.name == 'main' and record.levelname == 'ERROR' and \
           "Intentional test error occurred" in record.message and \
           record.exc_info and "Test error for Sentry" in str(record.exc_info[1]):
            assert hasattr(record, 'request_id'), "Error LogRecord missing 'request_id'"
            assert record.request_id == request_id_from_header
            assert hasattr(record, 'user_id'), "Error LogRecord missing 'user_id'"
            assert record.user_id is None
            assert isinstance(record.exc_info[1], ValueError)
            assert str(record.exc_info[1]) == "Test error for Sentry"
            found_record_attributes = True
            break
    assert found_record_attributes, "Target error LogRecord with correct attributes not found."

def test_user_id_logging_unauthenticated_request(caplog):
    caplog.set_level(logging.INFO)
    response = client.get("/health")
    assert response.status_code in [200, 404]

    request_id_from_header = response.headers.get("X-Request-ID")
    assert request_id_from_header is not None

    found_record_attributes = False
    for record in caplog.records:
        if record.name == 'main' and record.levelname == "INFO" and "Processing /health endpoint." in record.message:
            assert hasattr(record, 'request_id'), "LogRecord missing 'request_id' attribute"
            assert record.request_id == request_id_from_header, "LogRecord 'request_id' does not match header"
            assert hasattr(record, 'user_id'), "LogRecord missing 'user_id' attribute"
            assert record.user_id is None, f"Expected user_id to be None for unauthenticated /health request, got '{record.user_id}'"
            found_record_attributes = True
            break
    assert found_record_attributes, "Target LogRecord for unauthenticated /health request with correct attributes not found."

def test_user_id_logging_authenticated_request(caplog, test_user_token):
    """Test that user_id is logged for an authenticated request."""
    caplog.set_level(logging.INFO)
    headers = {"Authorization": f"Bearer {test_user_token['access_token']}"}
    response = client.get("/api/users/me", headers=headers)

    assert response.status_code == 403, f"Request to /api/users/me returned {response.status_code}, expected 403 due to unverified email. Content: {response.text}"

    authenticated_user_id = test_user_token['user_id']
    request_id_from_header = response.headers.get("X-Request-ID")
    assert request_id_from_header is not None

    auth_log_found = False
    for record in caplog.records:
        if getattr(record, 'request_id', None) == request_id_from_header:
            if getattr(record, 'user_id', None) == authenticated_user_id:
                # We are looking for *any* log that might have been emitted during this request
                # by our application that has the correct request_id and user_id.
                # It might not be from record.name == 'main' if, for example, the 403 is raised
                # very early by a dependency like fastapi.security.oauth2 before app-level logging.
                # The RequestIdMiddleware should still have processed the user_id for any log.
                print(f"Found potential authenticated log record: name='{record.name}', message='{record.message}', user_id='{record.user_id}'") # DEBUG
                auth_log_found = True
                break

    assert auth_log_found, f"Log record for authenticated request with user_id {authenticated_user_id} and request_id {request_id_from_header} not found. Relevant logs in caplog.text: \n{caplog.text}"

# Reminder for password logging prevention (as a comment)
# "REMINDER: Ensure that UserCreate, UserLogin, or any other models/dicts containing
# plain-text passwords are not logged directly. If they must be logged for debugging,
# ensure password fields are explicitly redacted or excluded."
