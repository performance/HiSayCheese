# tests/test_auth.py
import pytest
import uuid
from jose import jwt
from datetime import datetime, timedelta
from unittest.mock import patch

# --- Module Imports ---
# These are the things we need to import for our tests to run.
# We import from our application's modules.
from config import SECRET_KEY, ALGORITHM, FRONTEND_URL
from models.models import User as UserModel
from auth_utils import verify_password
from .conftest import TestingSessionLocal # Used to verify DB state independently.


# =====================================================================================
# ==                            REGISTRATION TESTS                                   ==
# =====================================================================================

def test_register_user_success(client):
    """
    GIVEN: A valid unique email and a strong password.
    WHEN: The /api/auth/register endpoint is called.
    THEN: It should return a 201 status code and the new user's data (without the password).
    AND: The user should be correctly created in the database, unverified, with a verification token.
    """
    unique_email = f"test_success_{uuid.uuid4()}@example.com"
    password = "vAl!dpassword123"

    # --- 1. ACTION: Call the API Endpoint ---
    response = client.post(
        "/api/auth/register",
        json={"email": unique_email, "password": password},
    )

    # --- 2. ASSERT: Check the HTTP Response ---
    assert response.status_code == 201, f"API call failed: {response.text}"
    data = response.json()
    assert data["email"] == unique_email
    assert "id" in data
    assert "hashed_password" not in data, "Security risk: Hashed password should never be in an API response."

    # --- 3. VERIFY: Check the Database State ---
    db = TestingSessionLocal()
    try:
        user_in_db = db.query(UserModel).filter(UserModel.email == unique_email).first()
        assert user_in_db is not None, "User was not created in the database."
        assert user_in_db.is_verified is False
        assert user_in_db.verification_token is not None
        assert verify_password(password, user_in_db.hashed_password)
    finally:
        db.close()

def test_register_user_duplicate_email(client):
    """
    GIVEN: An email address that already exists in the database.
    WHEN: The /api/auth/register endpoint is called with that email.
    THEN: It should return a 409 Conflict status code.
    """
    unique_email = f"test_duplicate_{uuid.uuid4()}@example.com"
    password = "vAl!dpassword123"

    # First, create the user successfully.
    client.post("/api/auth/register", json={"email": unique_email, "password": password})

    # Then, attempt to register again with the same email.
    response = client.post(
        "/api/auth/register",
        json={"email": unique_email, "password": password},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Email already registered"

@pytest.mark.parametrize(
    "password, expected_msg_part",
    [
        ("short", "at least 8 characters"),
        ("nouppercase1!", "an uppercase letter"),
        ("NOLOWERCASE1!", "a lowercase letter"),
        ("NoLoWeRcAsE!", "a digit"),
        ("NoLoWeRcAsE1", "a special character"),
    ],
)
def test_register_user_password_strength_violations(client, password, expected_msg_part):
    """
    GIVEN: A password that violates one of the strength rules.
    WHEN: The /api/auth/register endpoint is called.
    THEN: It should return a 422 status code with a descriptive error message.
    """
    response = client.post(
        "/api/auth/register",
        json={"email": f"pw_strength_{uuid.uuid4()}@example.com", "password": password},
    )
    assert response.status_code == 422
    # Check that the detailed error message contains the expected reason.
    assert expected_msg_part in response.text


# =====================================================================================
# ==                                 LOGIN TESTS                                     ==
# =====================================================================================

def test_login_success(client):
    """
    GIVEN: A registered user.
    WHEN: The /api/auth/login endpoint is called with correct credentials.
    THEN: It should return a 200 status code with an access token.
    AND: The token should contain the correct user information.
    """
    user_email = f"test_login_success_{uuid.uuid4()}@example.com"
    password = "vAl!dpassword123"

    # 1. Setup: Register the user. We patch the email service as we don't need to test it here.
    with patch("services.email_service.EmailService.send_email"):
        reg_response = client.post("/api/auth/register", json={"email": user_email, "password": password})
    assert reg_response.status_code == 201, "Test setup failed: could not register user."
    user_id_from_reg = reg_response.json()["id"]

    # 2. Action: Log in with the new credentials.
    login_response = client.post(
        "/api/auth/login",
        data={"username": user_email, "password": password}, # Form data
    )
    
    # 3. Assert: Check the login response and the token.
    assert login_response.status_code == 200, login_response.text
    token_data = login_response.json()
    assert "access_token" in token_data
    access_token = token_data["access_token"]
    assert token_data["token_type"] == "bearer"
    
    # 4. Verify: Decode the token to check its contents.
    decoded_token = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
    assert decoded_token["user_id"] == user_id_from_reg
    # assert decoded_token["exp"] > datetime.utcnow().timestamp()


def test_login_incorrect_password(client):
    """
    GIVEN: A registered user.
    WHEN: The /api/auth/login endpoint is called with the correct email but wrong password.
    THEN: It should return a 401 Unauthorized status code.
    """
    user_email = f"test_wrong_pw_{uuid.uuid4()}@example.com"
    password = "vAl!dpassword123"

    # Setup: Register the user.
    with patch("services.email_service.EmailService.send_email"):
        client.post("/api/auth/register", json={"email": user_email, "password": password})

    # Action: Attempt login with the wrong password.
    response = client.post(
        "/api/auth/login",
        data={"username": user_email, "password": "thisIsTheWrongPassword"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect email or password"


def test_login_nonexistent_user(client):
    """
    GIVEN: An email address that is not registered.
    WHEN: The /api/auth/login endpoint is called with that email.
    THEN: It should return a 401 Unauthorized status code to prevent user enumeration.
    """
    response = client.post(
        "/api/auth/login",
        data={"username": "nosuchuser@example.com", "password": "anypassword"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect email or password"

# =====================================================================================
# ==                         EMAIL VERIFICATION TESTS                                ==
# =====================================================================================

def test_email_verification_flow(client):
    """
    GIVEN: A new user is registered.
    WHEN: The verification link is 'clicked' (by calling the /verify-email endpoint with the token).
    THEN: The user's `is_verified` status should become True.
    AND: Subsequent attempts to use the same token should fail.
    """
    user_email = f"verify_flow_{uuid.uuid4()}@example.com"
    password = "vAl!dpassword123"

    # --- 1. Register User ---
    # We patch the email service to prevent actual email sending during tests.
    with patch("services.email_service.EmailService.send_email"):
        reg_response = client.post("/api/auth/register", json={"email": user_email, "password": password})
    assert reg_response.status_code == 201

    # --- 2. Get Verification Token from Database ---
    db = TestingSessionLocal()
    try:
        user_in_db = db.query(UserModel).filter(UserModel.email == user_email).first()
        assert user_in_db is not None
        assert user_in_db.is_verified is False
        verification_token = user_in_db.verification_token
        assert verification_token is not None
    finally:
        db.close()

    # --- 3. Action: Call the verification endpoint ---
    verify_response = client.get(f"/api/auth/verify-email?token={verification_token}")
    
    # --- 4. Assert: Check the verification response ---
    assert verify_response.status_code == 200, verify_response.text
    assert verify_response.json()["is_verified"] is True

    # --- 5. Verify: Check the database state again ---
    db = TestingSessionLocal()
    try:
        user_in_db_after = db.query(UserModel).filter(UserModel.email == user_email).first()
        assert user_in_db_after.is_verified is True
        assert user_in_db_after.verification_token is None, "Token should be cleared after successful verification."
    finally:
        db.close()
        
    # --- 6. Edge Case: Try to use the same token again ---
    reuse_response = client.get(f"/api/auth/verify-email?token={verification_token}")
    assert reuse_response.status_code == 400, "Reusing a verification token should fail."

def test_access_protected_route_unverified_vs_verified(client):
    """
    GIVEN: A registered but unverified user.
    WHEN: They try to access a protected route.
    THEN: They should be denied with a 403 Forbidden error.
    WHEN: They verify their email and try again with the same token.
    THEN: They should be granted access.
    """
    user_email = f"access_ctrl_{uuid.uuid4()}@example.com"
    password = "vAl!dpassword123"

    # 1. Register user
    with patch("services.email_service.EmailService.send_email"):
        client.post("/api/auth/register", json={"email": user_email, "password": password})
    
    # 2. Get login token for the unverified user
    login_response = client.post("/api/auth/login", data={"username": user_email, "password": password})
    access_token = login_response.json()["access_token"]
    auth_headers = {"Authorization": f"Bearer {access_token}"}
    
    # 3. Action: Attempt to access a protected route (/api/users/me)
    # This endpoint is protected by `get_current_user`, which checks `is_verified`.
    me_response_unverified = client.get("/api/users/me", headers=auth_headers)
    
    # 4. Assert: Access should be forbidden
    assert me_response_unverified.status_code == 403
    assert "Email not verified" in me_response_unverified.json()["detail"]

    # 5. Verify the user's email
    db = TestingSessionLocal()
    try:
        user_in_db = db.query(UserModel).filter(UserModel.email == user_email).first()
        verification_token = user_in_db.verification_token
    finally:
        db.close()
    
    client.get(f"/api/auth/verify-email?token={verification_token}") # "Click" the link
    
    # 6. Action: Try accessing the protected route again with the SAME token
    me_response_verified = client.get("/api/users/me", headers=auth_headers)

    # 7. Assert: Access should now be granted
    assert me_response_verified.status_code == 200
    assert me_response_verified.json()["email"] == user_email