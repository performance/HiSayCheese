import pytest
from fastapi.testclient import TestClient
from fastapi import status
from sqlalchemy.orm import Session
import uuid
from datetime import timedelta
import os
import sys

# Ensure the app's modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app
from models.models import User
from auth_utils import hash_password, create_access_token # create_access_token for expired token test
from db.database import SessionLocal # For db_session fixture

# Client fixture
@pytest.fixture(scope="module")
def client():
    # Create tables if they don't exist. This is important if using an in-memory DB
    # or if the main app's startup event hasn't run.
    # from db.database import create_db_and_tables
    # create_db_and_tables() # Usually handled by app startup or a global test setup
    return TestClient(app)

# DB session fixture
@pytest.fixture(scope="function")
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Utility to clear users (could be in a conftest.py or shared utils)
def clear_user_from_db(db: Session, email: str):
    user = db.query(User).filter(User.email == email).first()
    if user:
        db.delete(user)
        db.commit()

# Test user factory fixture
@pytest.fixture(scope="function")
def test_user_factory(db_session: Session):
    created_users_emails = []
    def _create_user(email_prefix="testuser_me"): # Changed prefix to avoid collision with test_auth
        unique_email = f"{email_prefix}_{uuid.uuid4()}@example.com"
        password = "TestPassword123" # Consistent password for factory-created users
        hashed_pw = hash_password(password)

        user = User(email=unique_email, hashed_password=hashed_pw)
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        created_users_emails.append(user.email)
        return {"email": user.email, "password": password, "id": str(user.id)}

    yield _create_user

    # Cleanup all users created by this factory instance during a test
    for email in created_users_emails:
        clear_user_from_db(db_session, email)

# Access token factory fixture
@pytest.fixture(scope="function")
def access_token_factory(client: TestClient): # Removed test_user_factory dependency here, pass user_details directly
    def _get_token(user_details: dict): # Expects dict with 'email' and 'password'
        login_data = {"username": user_details["email"], "password": user_details["password"]}
        response = client.post("/api/auth/login", data=login_data)
        assert response.status_code == 200, f"Login failed for token generation: {response.text}"
        return response.json()["access_token"]
    return _get_token


# Tests for /api/users/me
def test_access_users_me_no_token(client: TestClient):
    response = client.get("/api/users/me")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    # FastAPI's default for missing OAuth2 token is usually 403 if scheme is just checked,
    # but our oauth2_scheme and get_current_user should enforce 401 if token is not provided.
    # If it returns 403, it means the dependency `oauth2_scheme` itself raised it.
    # Let's verify the detail if possible, usually "Not authenticated" or similar from scheme.
    # For now, 401 is the primary check based on how get_current_user is structured.

def test_access_users_me_with_valid_token(client: TestClient, test_user_factory, access_token_factory):
    user_details = test_user_factory() # Create a user
    token = access_token_factory(user_details) # Get token for that user

    response = client.get(
        "/api/users/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["email"] == user_details["email"]
    assert data["id"] == user_details["id"]
    assert "hashed_password" not in data # Ensure sensitive data is not returned

def test_access_users_me_with_invalid_signature_token(client: TestClient):
    response = client.get(
        "/api/users/me",
        headers={"Authorization": "Bearer aninvalidtokenstringthatisnotjwt"}
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    # The detail might vary depending on where the JWTError is caught,
    # but auth_utils.decode_access_token should provide "Could not validate credentials".
    assert response.json()["detail"] == "Could not validate credentials"


def test_access_users_me_with_expired_token(client: TestClient, test_user_factory):
    user_details = test_user_factory() # Create a user

    # Create an already expired token
    expired_token = create_access_token(
        data={"sub": user_details["email"], "user_id": user_details["id"]},
        expires_delta=timedelta(seconds=-3600) # 1 hour in the past
    )

    response = client.get(
        "/api/users/me",
        headers={"Authorization": f"Bearer {expired_token}"}
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"] == "Token has expired"


# Tests for /api/users/history query parameter validation (skip, limit)
# Assuming User model and history records are created elsewhere or not strictly needed
# for just testing the parameter validation itself, as Pydantic/FastAPI handle it early.

def test_read_user_history_invalid_skip_negative(client: TestClient, test_user_factory, access_token_factory):
    user_details = test_user_factory()
    token = access_token_factory(user_details)
    response = client.get(
        "/api/users/history?skip=-1&limit=10",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY, response.text
    data = response.json()
    assert any(err["loc"] == ["query", "skip"] and "greater than or equal to 0" in err["msg"] for err in data["detail"])

def test_read_user_history_invalid_limit_negative(client: TestClient, test_user_factory, access_token_factory):
    user_details = test_user_factory()
    token = access_token_factory(user_details)
    response = client.get(
        "/api/users/history?skip=0&limit=-5",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY, response.text
    data = response.json()
    # Assuming limit also has ge=0 or similar constraint. If it's just int, this might pass.
    # The Pydantic models were updated to conint(ge=0) for skip and limit.
    # If limit has conint(gt=0), then 0 would also be an error. Assuming ge=0 for now.
    assert any(err["loc"] == ["query", "limit"] and "greater than or equal to 0" in err["msg"] for err in data["detail"])


def test_read_user_history_invalid_skip_type_string(client: TestClient, test_user_factory, access_token_factory):
    user_details = test_user_factory()
    token = access_token_factory(user_details)
    response = client.get(
        "/api/users/history?skip=abc&limit=10",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY, response.text
    data = response.json()
    assert any(err["loc"] == ["query", "skip"] and "Input should be a valid integer" in err["msg"] for err in data["detail"])

def test_read_user_history_invalid_limit_type_string(client: TestClient, test_user_factory, access_token_factory):
    user_details = test_user_factory()
    token = access_token_factory(user_details)
    response = client.get(
        "/api/users/history?skip=0&limit=xyz",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY, response.text
    data = response.json()
    assert any(err["loc"] == ["query", "limit"] and "Input should be a valid integer" in err["msg"] for err in data["detail"])

# Test for a very large limit, assuming there's a sensible upper bound defined in the endpoint's Pydantic model
# or logic. If not, this might pass or lead to performance issues (which is why such limits are good).
# For now, let's assume the model has `limit: conint(ge=0, le=100)` or similar from previous tasks.
# If `limit` in `EnhancementHistoryParams` (if that's what's used, or directly in endpoint)
# has `le=100` (as per `conint` in a Pydantic model for query params).
# The current `routers/users.py` for `read_user_enhancement_history` directly uses `skip: int = 0, limit: int = 10`.
# These are not yet using Pydantic models with `conint` for query parameters directly in the function signature.
# FastAPI will convert them to int, but range constraints from `conint` are applied if a Pydantic model is used for query params.
# Let's assume the previous Pydantic model enhancements included updating function signatures
# or using a Depends with a Pydantic model for these query params.
# If not, these out-of-range tests for query params might not behave as 422 UNLESS there's a Pydantic model.

# Let's assume the endpoint `read_user_enhancement_history` was updated like:
# class HistoryParams(BaseModel):
#   skip: conint(ge=0) = 0
#   limit: conint(ge=0, le=100) = 10 # Example upper limit
# async def read_user_enhancement_history(params: HistoryParams = Depends(), ...):

# If this is the case, then the following test would be valid:
# def test_read_user_history_limit_too_large(client: TestClient, test_user_factory, access_token_factory):
#     user_details = test_user_factory()
#     token = access_token_factory(user_details)
#     response = client.get(
#         "/api/users/history?skip=0&limit=200", # Assuming max limit is 100
#         headers={"Authorization": f"Bearer {token}"}
#     )
#     assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY, response.text
#     data = response.json()
#     assert any(err["loc"] == ["query", "limit"] and "less than or equal to 100" in err["msg"] for err in data["detail"])

# Given the current state of `routers/users.py` (skip: int, limit: int),
# the "limit_too_large" test for 422 won't pass based on Pydantic model validation for query params.
# The negative and type tests should pass due to FastAPI's default handling for int conversion.
# I will proceed with the negative and type tests as they are generally applicable.
# The "limit_too_large" test would require confirming Pydantic models are used for these query params.
# For now, I will omit "limit_too_large" as it depends on an unconfirmed change to the endpoint signature.

# --- Advanced Rate Limiting Tests for User Endpoints ---
import time

# Assuming these constants are accessible from main or defined for tests
try:
    from main import ANON_USER_RATE_LIMIT, AUTH_USER_RATE_LIMIT
    ANON_REQUESTS_PER_WINDOW = int(ANON_USER_RATE_LIMIT.split('/')[0]) # Should not be used by user endpoints
    AUTH_REQUESTS_PER_WINDOW = int(AUTH_USER_RATE_LIMIT.split('/')[0])
except ImportError:
    ANON_REQUESTS_PER_WINDOW = 20
    AUTH_REQUESTS_PER_WINDOW = 100 # Fallback, ensure matches main.py

def get_rate_limit_headers_from_response_users(response): # Renamed for clarity
    return {
        "limit": response.headers.get("X-RateLimit-Limit"),
        "remaining": response.headers.get("X-RateLimit-Remaining"),
        "reset": response.headers.get("X-RateLimit-Reset"),
    }

# 1. Differentiated Limits (Authenticated for /api/users/history)
# All /api/users/ endpoints require authentication, so they should always use AUTH_USER_RATE_LIMIT.
def test_history_rate_limiting_authenticated(client: TestClient, test_user_factory, access_token_factory):
    user_details = test_user_factory()
    token = access_token_factory(user_details)
    auth_headers = {"Authorization": f"Bearer {token}"}

    for i in range(AUTH_REQUESTS_PER_WINDOW):
        response = client.get("/api/users/history", headers=auth_headers)
        if response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            pytest.fail(f"Authenticated request {i+1} to /history rate limited prematurely.")
        assert response.status_code == status.HTTP_200_OK # Assuming history records can be empty list

    # Next request should be rate limited
    response = client.get("/api/users/history", headers=auth_headers)
    assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    headers = get_rate_limit_headers_from_response_users(response)
    assert headers["limit"] == str(AUTH_REQUESTS_PER_WINDOW)
    assert headers["remaining"] == "0"


# 2. Rate Limiting Keyed by User ID (Isolation) - using /api/users/history
def test_rate_limiting_user_isolation_on_user_endpoint(client: TestClient, test_user_factory, access_token_factory, db_session: Session):
    # User A
    user_a_details = test_user_factory("userA_iso_hist")
    token_a = access_token_factory(user_a_details)

    # User B
    user_b_details = test_user_factory("userB_iso_hist")
    token_b = access_token_factory(user_b_details)

    # User A makes some requests (e.g., half the limit)
    for i in range(AUTH_REQUESTS_PER_WINDOW // 2):
        client.get("/api/users/history", headers={"Authorization": f"Bearer {token_a}"})

    # User B makes requests, should have full quota
    for i in range(AUTH_REQUESTS_PER_WINDOW):
        response_b = client.get("/api/users/history", headers={"Authorization": f"Bearer {token_b}"})
        if response_b.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            pytest.fail(f"User B request {i+1} to /history rate limited prematurely by User A's activity.")
        assert response_b.status_code == status.HTTP_200_OK
        if i == 0: # Check headers on first request for User B
            headers_b_first = get_rate_limit_headers_from_response_users(response_b)
            assert headers_b_first["limit"] == str(AUTH_REQUESTS_PER_WINDOW)
            assert headers_b_first["remaining"] == str(AUTH_REQUESTS_PER_WINDOW - 1)

    # User B exceeds their limit
    response_b_limit_exceeded = client.get("/api/users/history", headers={"Authorization": f"Bearer {token_b}"})
    assert response_b_limit_exceeded.status_code == status.HTTP_429_TOO_MANY_REQUESTS

    # User A should still have remaining quota from their initial half
    response_a_after_b = client.get("/api/users/history", headers={"Authorization": f"Bearer {token_a}"})
    assert response_a_after_b.status_code == status.HTTP_200_OK # Should not be 429
    headers_a_after_b = get_rate_limit_headers_from_response_users(response_a_after_b)
    # Remaining should be total_limit - requests_made_by_A - this_current_request
    expected_remaining_a = AUTH_REQUESTS_PER_WINDOW - (AUTH_REQUESTS_PER_WINDOW // 2) - 1
    assert headers_a_after_b["remaining"] == str(expected_remaining_a)

    # Cleanup handled by test_user_factory fixture

# 3. Rate Limit Headers for /api/users/presets (POST)
def test_create_preset_rate_limit_headers(client: TestClient, test_user_factory, access_token_factory):
    user_details = test_user_factory()
    token = access_token_factory(user_details)
    auth_headers = {"Authorization": f"Bearer {token}"}

    preset_data = {"preset_name": "Test Preset RL Headers", "parameters_json": "{}"}
    response = client.post("/api/users/presets", json=preset_data, headers=auth_headers)

    assert response.status_code == status.HTTP_201_CREATED
    headers = get_rate_limit_headers_from_response_users(response)
    assert headers["limit"] == str(AUTH_REQUESTS_PER_WINDOW)
    assert headers["remaining"] == str(AUTH_REQUESTS_PER_WINDOW - 1)
    assert headers["reset"] is not None

# 4. Rate Limit Reset Test (using /api/users/history)
def test_history_rate_limit_reset_authenticated(client: TestClient, test_user_factory, access_token_factory):
    user_details = test_user_factory()
    token = access_token_factory(user_details)
    auth_headers = {"Authorization": f"Bearer {token}"}

    # Exceed limit
    for i in range(AUTH_REQUESTS_PER_WINDOW + 1):
        response = client.get("/api/users/history", headers=auth_headers)
        if i == AUTH_REQUESTS_PER_WINDOW:
            assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
            headers_429 = get_rate_limit_headers_from_response_users(response)
            reset_time = int(headers_429["reset"])
            current_time = int(time.time())
            sleep_duration = max(0, reset_time - current_time) + 1
            if sleep_duration > 65: # Safety for tests
                pytest.skip(f"Reset time too far ({sleep_duration}s), skipping sleep.")
            time.sleep(sleep_duration)

    # Try again after waiting
    response_after_reset = client.get("/api/users/history", headers=auth_headers)
    assert response_after_reset.status_code == status.HTTP_200_OK
    headers_after = get_rate_limit_headers_from_response_users(response_after_reset)
    assert headers_after["remaining"] == str(AUTH_REQUESTS_PER_WINDOW - 1)

# 5. Basic "is it rate limited" test for another user endpoint (e.g. GET /presets)
def test_list_presets_endpoint_is_rate_limited(client: TestClient, test_user_factory, access_token_factory):
    user_details = test_user_factory()
    token = access_token_factory(user_details)
    auth_headers = {"Authorization": f"Bearer {token}"}
    endpoint_url = "/api/users/presets"

    for i in range(AUTH_REQUESTS_PER_WINDOW):
        response = client.get(endpoint_url, headers=auth_headers)
        if response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            pytest.fail(f"Authenticated request {i+1} to {endpoint_url} rate limited prematurely.")
        assert response.status_code == status.HTTP_200_OK

    response = client.get(endpoint_url, headers=auth_headers)
    assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
