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
