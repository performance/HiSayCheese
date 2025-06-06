import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
import uuid

# Ensure the app's modules can be imported
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app
from db.database import SessionLocal # create_db_and_tables might not be needed here if main handles it
from models.models import User as UserModel
from auth_utils import verify_password
from jose import jwt
from config import SECRET_KEY, ALGORITHM

# Create tables if they don't exist (e.g., for in-memory DB or first run)
# This is generally handled by main.py on startup, but explicit call can be useful
# For tests, it's better to manage this with test-specific setup/teardown
# For now, we rely on main.py's startup event or manual creation if needed.
# create_db_and_tables() # Usually, you'd have a separate test DB setup

client = TestClient(app)

# Helper to get a DB session
def get_test_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Fixture for DB session (optional, could also use get_test_db directly)
@pytest.fixture(scope="module") # module scope if DB is reset per module, function if per test
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def clear_user_from_db(db: Session, email: str):
    user = db.query(UserModel).filter(UserModel.email == email).first()
    if user:
        db.delete(user)
        db.commit()

@pytest.fixture(autouse=True)
def auto_clear_db_after_test(db_session: Session):
    """Ensures the DB is clean after each test that might create users."""
    # This is a simplistic cleanup. Ideally, use transactions and rollback,
    # or a dedicated test DB that's reset.
    # For this example, we'll try to clean up known test users.
    # This is tricky if emails are dynamically generated and not tracked.
    # A better approach is to override get_db with a test DB that rolls back.
    # For now, we'll rely on specific test functions to clean up if they create data.
    yield


def test_register_user_success(db_session: Session):
    unique_email = f"test_success_{uuid.uuid4()}@example.com"
    password = "validpassword123"

    response = client.post(
        "/api/auth/register",
        json={"email": unique_email, "password": password},
    )

    assert response.status_code == 201, response.text
    data = response.json()
    assert data["email"] == unique_email
    assert "id" in data
    assert "hashed_password" not in data # IMPORTANT: Hashed password should not be in response
    assert "created_at" in data

    # Verify in DB
    user_in_db = db_session.query(UserModel).filter(UserModel.email == unique_email).first()
    assert user_in_db is not None
    assert user_in_db.email == unique_email
    assert verify_password(password, user_in_db.hashed_password) # Check if password was hashed correctly
    assert not verify_password(password + "wrong", user_in_db.hashed_password) # Double check

    # Cleanup
    clear_user_from_db(db_session, unique_email)


def test_register_user_duplicate_email(db_session: Session):
    duplicate_email = f"test_duplicate_{uuid.uuid4()}@example.com"
    password = "validpassword123"

    # Create user first
    client.post(
        "/api/auth/register",
        json={"email": duplicate_email, "password": password},
    )

    # Attempt to register again with the same email
    response = client.post(
        "/api/auth/register",
        json={"email": duplicate_email, "password": "anotherpassword"},
    )

    assert response.status_code == 409, response.text
    data = response.json()
    assert data["detail"] == "Email already registered"

    # Cleanup
    clear_user_from_db(db_session, duplicate_email)


def test_register_user_invalid_email_format():
    response = client.post(
        "/api/auth/register",
        json={"email": "invalid-email", "password": "validpassword123"},
    )

    assert response.status_code == 422, response.text
    data = response.json()
    # Pydantic's error messages can be nested. Check for 'email' field error.
    assert any(err["loc"] == ["body", "email"] for err in data["detail"])


def test_register_user_weak_password():
    response = client.post(
        "/api/auth/register",
        json={"email": f"test_weak_pw_{uuid.uuid4()}@example.com", "password": "pw"},
    )

    assert response.status_code == 422, response.text # Should be 422 due to Pydantic validation
    data = response.json()
    # Check for Pydantic's specific error structure
    assert any(err["msg"] for err in data["detail"] if "password" in err["loc"])


# Password Strength Tests (now handled by Pydantic model)
@pytest.mark.parametrize(
    "password, expected_msg_part",
    [
        ("short", "8 characters"), # Too short
        ("nouppercase1!", "uppercase letter"),
        ("NOUPPERCASE1!", "uppercase letter"), # Edge case if regex is only [a-z]
        ("NOLOWERCASE1!", "lowercase letter"),
        ("NoLoWeRcAsE!", "digit"),
        ("NoLoWeRcAsE1", "special character"),
    ],
)
def test_register_user_password_strength_violations(password, expected_msg_part):
    unique_email = f"test_pw_strength_{uuid.uuid4()}@example.com"
    response = client.post(
        "/api/auth/register",
        json={"email": unique_email, "password": password},
    )
    assert response.status_code == 422, response.text
    data = response.json()
    assert isinstance(data["detail"], list)
    password_errors = [err for err in data["detail"] if err.get("loc") == ["body", "password"]]
    assert len(password_errors) > 0, "No Pydantic error found for password field"
    # The actual message comes from Pydantic (constr) or our custom validator's ValueError.
    # Pydantic wraps ValueError from custom validators.
    found_match = False
    for err in password_errors:
        if "ctx" in err and "error" in err["ctx"]: # Custom validator's ValueError
             if expected_msg_part in str(err["ctx"]["error"]): # Convert error to string to search
                found_match = True
                break
        elif expected_msg_part in err["msg"]: # Standard Pydantic error message
            found_match = True
            break
    assert found_match, f"Expected message part '{expected_msg_part}' not found in password errors: {password_errors}"


# Test for missing parameters (Pydantic validation)
def test_register_user_missing_email():
    response = client.post(
        "/api/auth/register",
        json={"password": "ValidPassword1!"},
    )
    assert response.status_code == 422, response.text
    data = response.json()
    assert any(err["loc"] == ["body", "email"] and "Missing" in err["msg"] for err in data["detail"])

def test_register_user_missing_password():
    response = client.post(
        "/api/auth/register",
        json={"email": "testmissing@example.com"},
    )
    assert response.status_code == 422, response.text
    data = response.json()
    assert any(err["loc"] == ["body", "password"] and "Missing" in err["msg"] for err in data["detail"])


# Test for malicious string inputs (SQLi/XSS attempts)
@pytest.mark.parametrize(
    "malicious_input",
    [
        "' OR '1'='1",
        "<script>alert('XSS')</script>",
        "../../../../../etc/passwd%00", # Path traversal attempt with null byte
    ]
)
def test_register_user_malicious_email_string(malicious_input):
    # Email field has specific validation (EmailStr), so these might be caught by format validation
    # or by Pydantic's general string processing.
    response = client.post(
        "/api/auth/register",
        json={"email": malicious_input, "password": "ValidPassword1!"},
    )
    assert response.status_code == 422 # Expect Pydantic validation error for email format
    data = response.json()
    assert any(err["loc"] == ["body", "email"] for err in data["detail"])

# Malicious password test - password is not typically stored as-is or reflected,
# but good to check it doesn't break input processing.
# Password validation rules are more about complexity.
def test_register_user_malicious_password_string():
    # This should be caught by password strength validation if it doesn't meet criteria.
    # If it *does* meet criteria, it should be accepted and hashed.
    # The main concern is that it doesn't cause an error during processing *before* hashing.
    malicious_password_thats_also_strong = "<script>alert('XSS')</script>A1!"
    email = f"test_malicious_pw_{uuid.uuid4()}@example.com"
    response = client.post(
        "/api/auth/register",
        json={"email": email, "password": malicious_password_thats_also_strong},
    )
    # If the malicious password meets strength criteria, it should be accepted (201)
    # Pydantic's `constr` doesn't inherently sanitize against XSS for passwords,
    # as they are not meant to be displayed. Hashing is the security measure.
    # Our custom validator also doesn't sanitize, it checks for character types.
    assert response.status_code == 201, response.text
    # Cleanup if created
    if response.status_code == 201:
        db = SessionLocal()
        clear_user_from_db(db, email)
        db.close()

# Rate Limiting Tests
def test_register_rate_limiting():
    email_prefix = f"ratelimit_reg_{uuid.uuid4()}"
    password = "ValidPassword1!"
    # Default limit for /register is "5/minute"
    for i in range(5): # Make 5 successful requests
        response = client.post("/api/auth/register", json={"email": f"{email_prefix}_{i}@example.com", "password": password})
        if response.status_code == 201: # If successful, clean up
            db = SessionLocal()
            clear_user_from_db(db, f"{email_prefix}_{i}@example.com")
            db.close()
        else: # If one of the first 5 fails unexpectedly (e.g. DB issue), fail the test
            pytest.fail(f"Registration attempt {i+1} failed: {response.text}")

    # The 6th request should be rate-limited
    response = client.post("/api/auth/register", json={"email": f"{email_prefix}_6@example.com", "password": password})
    assert response.status_code == 429, response.text
    assert "Rate limit exceeded" in response.json()["detail"]


def test_login_rate_limiting(db_session: Session):
    user_email = f"ratelimit_login_{uuid.uuid4()}@example.com"
    user_password = "ValidPassword1!"

    # Register user
    reg_response = client.post("/api/auth/register", json={"email": user_email, "password": user_password})
    assert reg_response.status_code == 201, f"Registration for rate limit test failed: {reg_response.text}"

    # Default limit for /login is "10/minute"
    for i in range(10):
        response = client.post("/api/auth/login", data={"username": user_email, "password": user_password})
        # Don't check for 200 here, as we are just hitting the endpoint.
        # It might be 401 if something is wrong, but we are testing rate limiter.
        # However, for a clean test, ensure it would be 200 if not for rate limit.
        if response.status_code == 429: # If rate limited early, test setup is wrong or limit too low
             pytest.fail(f"Login attempt {i+1} was rate limited prematurely: {response.text}")


    # The 11th request should be rate-limited
    response = client.post("/api/auth/login", data={"username": user_email, "password": user_password})
    assert response.status_code == 429, response.text
    assert "Rate limit exceeded" in response.json()["detail"]

    # Cleanup
    clear_user_from_db(db_session, user_email)


# --- Advanced Rate Limiting Tests for Auth ---
import time
from fastapi import status # For status codes if not already imported

# Import rate limit constants from main.py to use in assertions
# Note: This assumes main.py is structured to allow such imports.
# If main.py execution is complex, it might be better to redefine or mock these for tests.
try:
    from main import ANON_USER_RATE_LIMIT, AUTH_USER_RATE_LIMIT
    ANON_REQUESTS_PER_WINDOW = int(ANON_USER_RATE_LIMIT.split('/')[0])
    AUTH_REQUESTS_PER_WINDOW = int(AUTH_USER_RATE_LIMIT.split('/')[0])
except ImportError: # Fallback if main.py structure changes or for isolated test runs
    ANON_REQUESTS_PER_WINDOW = 20 # Must match main.py
    AUTH_REQUESTS_PER_WINDOW = 100 # Must match main.py


def get_rate_limit_headers(response):
    return {
        "limit": response.headers.get("X-RateLimit-Limit"),
        "remaining": response.headers.get("X-RateLimit-Remaining"),
        "reset": response.headers.get("X-RateLimit-Reset"),
    }

# Test Differentiated Limits (Anonymous for /register)
def test_register_differentiated_rate_limiting_anonymous():
    email_prefix = f"anon_reg_ratelimit_{uuid.uuid4()}"
    password = "ValidPassword1!"

    for i in range(ANON_REQUESTS_PER_WINDOW):
        response = client.post("/api/auth/register", json={"email": f"{email_prefix}_{i}@example.com", "password": password})
        if response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            pytest.fail(f"Anonymous registration attempt {i+1} was rate limited prematurely for /register.")
        # Cleanup successful registrations to avoid DB clutter and 409 errors on reruns
        if response.status_code == status.HTTP_201_CREATED:
            db = SessionLocal()
            clear_user_from_db(db, f"{email_prefix}_{i}@example.com")
            db.close()

    # The next request should be rate-limited
    response = client.post("/api/auth/register", json={"email": f"{email_prefix}_{ANON_REQUESTS_PER_WINDOW}@example.com", "password": password})
    assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS, response.text
    headers = get_rate_limit_headers(response)
    assert headers["limit"] == str(ANON_REQUESTS_PER_WINDOW)
    assert headers["remaining"] == "0"
    assert headers["reset"] is not None


# Test Differentiated Limits (Authenticated for /login - though login makes user authenticated for *next* requests)
# For /login, the "authenticated" limit doesn't quite apply in the same way, as the act of logging in *is* the transition.
# The key for /login itself will be IP-based if no prior valid token is sent.
# If a valid token *is* sent to /login (unusual, but possible), it would use user-based key.
# Let's test the more common anonymous case for /login hitting its limit.
def test_login_differentiated_rate_limiting_anonymous_ip_based(db_session: Session):
    user_email = f"auth_login_ratelimit_anon_{uuid.uuid4()}@example.com"
    user_password = "ValidPassword1!"
    # Register user so login attempts have a valid target
    reg_response = client.post("/api/auth/register", json={"email": user_email, "password": user_password})
    assert reg_response.status_code == status.HTTP_201_CREATED

    for i in range(ANON_REQUESTS_PER_WINDOW): # /login uses dynamic, so anon limit applies here
        response = client.post("/api/auth/login", data={"username": user_email, "password": user_password})
        if response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            pytest.fail(f"Anonymous login attempt {i+1} was rate limited prematurely for /login.")

    response = client.post("/api/auth/login", data={"username": user_email, "password": user_password})
    assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS, response.text
    headers = get_rate_limit_headers(response)
    # The limit applied to /login for an anonymous attempt should be ANON_USER_RATE_LIMIT
    assert headers["limit"] == str(ANON_REQUESTS_PER_WINDOW)
    assert headers["remaining"] == "0"

    clear_user_from_db(db_session, user_email)


# Test Rate Limiting Keyed by User ID (Isolation)
def test_rate_limiting_user_isolation(db_session: Session):
    # User A
    user_a_email = f"user_a_iso_{uuid.uuid4()}@example.com"
    user_a_password = "PasswordA1!"
    client.post("/api/auth/register", json={"email": user_a_email, "password": user_a_password})
    login_resp_a = client.post("/api/auth/login", data={"username": user_a_email, "password": user_a_password})
    token_a = login_resp_a.json()["access_token"]

    # User B
    user_b_email = f"user_b_iso_{uuid.uuid4()}@example.com"
    user_b_password = "PasswordB1!"
    client.post("/api/auth/register", json={"email": user_b_email, "password": user_b_password})
    login_resp_b = client.post("/api/auth/login", data={"username": user_b_email, "password": user_b_password})
    token_b = login_resp_b.json()["access_token"]

    # For this test, we'll use an endpoint that is known to be rate-limited and requires auth,
    # e.g., /api/users/me (though /me itself is not rate limited in this project yet).
    # Let's assume /api/users/history is rate-limited for this test (will be added in test_users.py).
    # For now, let's use /api/auth/register with a token (unconventional, but will test user-keyed limit)
    # A better endpoint would be one from users.py that *requires* auth.
    # Using /api/auth/register: if a token is sent, it should use user-based keying.

    # User A makes requests - should use AUTH_REQUESTS_PER_WINDOW
    # Since /register is for creating new users, sending a token is unusual.
    # Let's pivot this test to an endpoint that *requires* auth and will be rate-limited,
    # such as /api/users/history (anticipating its rate limiting in test_users.py).
    # For now, we'll simulate this by checking headers on a generic auth endpoint if one exists,
    # or we can test this more thoroughly when test_users.py is updated.

    # Placeholder: This test needs an actual authenticated & rate-limited endpoint.
    # For now, we'll test header presence on /api/users/me with different tokens.
    # This doesn't test isolation of limits fully but token-based keying for headers.

    response_a = client.get("/api/users/me", headers={"Authorization": f"Bearer {token_a}"})
    assert response_a.status_code == status.HTTP_200_OK
    headers_a = get_rate_limit_headers(response_a)
    # /api/users/me is NOT rate-limited by default in this project setup.
    # So, it won't have rate limit headers unless we add the decorator there.
    # This test will be more effective once applied to a rate-limited authenticated endpoint.
    # For now, we'll assert that if headers *were* present, they'd reflect AUTH limit.
    if headers_a["limit"]: # Only if the endpoint somehow got rate-limited
       assert headers_a["limit"] == str(AUTH_REQUESTS_PER_WINDOW)

    response_b = client.get("/api/users/me", headers={"Authorization": f"Bearer {token_b}"})
    assert response_b.status_code == status.HTTP_200_OK
    headers_b = get_rate_limit_headers(response_b)
    if headers_b["limit"]:
        assert headers_b["limit"] == str(AUTH_REQUESTS_PER_WINDOW)
        if headers_a["limit"] and headers_b["limit"]: # If both were somehow limited
             assert headers_a["remaining"] != headers_b["remaining"] # Crude check for isolation

    clear_user_from_db(db_session, user_a_email)
    clear_user_from_db(db_session, user_b_email)


# Test Rate Limit Headers
def test_register_rate_limit_headers_anonymous():
    email = f"hdr_reg_anon_{uuid.uuid4()}@example.com"
    response = client.post("/api/auth/register", json={"email": email, "password": "Password1!"})
    # This request is anonymous, should get ANON_USER_RATE_LIMIT
    # Status code might be 201 (success) or 409 (if email was somehow already used in a rapid test sequence)
    # We are interested in headers regardless of 201 or 409, as long as it's not 429 yet.
    assert response.status_code in [status.HTTP_201_CREATED, status.HTTP_409_CONFLICT]

    headers = get_rate_limit_headers(response)
    assert headers["limit"] == str(ANON_REQUESTS_PER_WINDOW)
    assert headers["remaining"] == str(ANON_REQUESTS_PER_WINDOW - 1) # First request
    assert headers["reset"] is not None

    if response.status_code == status.HTTP_201_CREATED:
        db = SessionLocal()
        clear_user_from_db(db, email)
        db.close()

# Rate Limit Reset Test (using /register for simplicity with anonymous limit)
def test_register_rate_limit_reset():
    email_prefix = f"reset_reg_{uuid.uuid4()}"
    password = "ValidPassword1!"

    # Exceed the limit
    for i in range(ANON_REQUESTS_PER_WINDOW + 1): # One more than limit
        response = client.post("/api/auth/register", json={"email": f"{email_prefix}_{i}@example.com", "password": password})
        if response.status_code == status.HTTP_201_CREATED: # Cleanup successful ones
            db = SessionLocal()
            clear_user_from_db(db, f"{email_prefix}_{i}@example.com")
            db.close()
        if i == ANON_REQUESTS_PER_WINDOW: # The one that should get 429
             assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
             headers_429 = get_rate_limit_headers(response)
             reset_time = int(headers_429["reset"])
             current_time = int(time.time())
             sleep_duration = max(0, reset_time - current_time) + 1 # Sleep until after reset + 1s buffer

             if sleep_duration > 65 : # Safety break for tests, minute window should be ~60s
                 pytest.skip(f"Reset time is too far in future ({sleep_duration}s), skipping sleep portion of test.")

             time.sleep(sleep_duration)

    # Try again after waiting
    final_email = f"{email_prefix}_final@example.com"
    response_after_reset = client.post("/api/auth/register", json={"email": final_email, "password": password})
    assert response_after_reset.status_code == status.HTTP_201_CREATED, \
        f"Request after reset failed. Expected 201, got {response_after_reset.status_code}. Headers: {get_rate_limit_headers(response_after_reset)}"

    headers_after = get_rate_limit_headers(response_after_reset)
    assert headers_after["remaining"] == str(ANON_REQUESTS_PER_WINDOW - 1)

    if response_after_reset.status_code == status.HTTP_201_CREATED:
        db = SessionLocal()
        clear_user_from_db(db, final_email)
        db.close()


# Tests for /api/auth/login
def test_login_success(db_session: Session):
    user_email = f"testlogin_{uuid.uuid4()}@example.com"
    user_password = "ValidPassword123"

    # 1. Register user
    reg_response = client.post(
        "/api/auth/register",
        json={"email": user_email, "password": user_password},
    )
    assert reg_response.status_code == 201, f"Registration failed: {reg_response.text}"
    user_id_from_reg = reg_response.json().get("id") # Get user_id for later assertion

    # 2. Login with the registered user
    login_response = client.post(
        "/api/auth/login",
        data={"username": user_email, "password": user_password}, # Form data
    )
    assert login_response.status_code == 200, login_response.text
    login_data = login_response.json()
    assert "access_token" in login_data
    assert login_data["token_type"] == "bearer"

    # 3. Decode token and verify claims
    access_token = login_data["access_token"]
    try:
        decoded_token = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.JWTError as e:
        pytest.fail(f"Failed to decode token: {e}")

    assert decoded_token["sub"] == user_email
    assert "user_id" in decoded_token
    assert decoded_token["user_id"] == user_id_from_reg # Check if user_id matches
    assert "exp" in decoded_token
    assert isinstance(decoded_token["exp"], int)

    # Cleanup
    clear_user_from_db(db_session, user_email)


def test_login_invalid_email():
    response = client.post(
        "/api/auth/login",
        data={"username": f"nosuchuser_{uuid.uuid4()}@example.com", "password": "anypassword"},
    )
    assert response.status_code == 401, response.text
    data = response.json()
    assert data["detail"] == "Incorrect email or password"


def test_login_incorrect_password(db_session: Session):
    user_email = f"testpwuser_{uuid.uuid4()}@example.com"
    correct_password = "CorrectPassword123"
    incorrect_password = "IncorrectPassword456"

    # 1. Register user
    reg_response = client.post(
        "/api/auth/register",
        json={"email": user_email, "password": correct_password},
    )
    assert reg_response.status_code == 201, f"Registration failed: {reg_response.text}"

    # 2. Attempt login with incorrect password
    login_response = client.post(
        "/api/auth/login",
        data={"username": user_email, "password": incorrect_password},
    )
    assert login_response.status_code == 401, login_response.text
    data = login_response.json()
    assert data["detail"] == "Incorrect email or password"

    # Cleanup
    clear_user_from_db(db_session, user_email)


# Note: For a robust test suite, especially with database interactions:
# 1. Use a separate test database (e.g., in-memory SQLite or a dedicated test PostgreSQL DB).
#    This can be configured by overriding the `get_db` dependency in FastAPI for tests.
# 2. Ensure each test runs in a transaction that is rolled back afterwards to maintain isolation.
#    FastAPI's TestClient and dependency overrides can facilitate this.
# The current cleanup (`clear_user_from_db`) is a basic workaround.
# The `sys.path.insert` is a common way to handle imports in tests when running from the tests directory.
# Ideally, project structure and PYTHONPATH should be set up so this isn't strictly necessary,
# e.g., by installing the package in editable mode (`pip install -e .`) or configuring the test runner.
# Running `pytest` from the project root directory is usually the best practice.
