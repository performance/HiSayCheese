import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
import uuid

# Ensure the app's modules can be imported
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app
from db.database import SessionLocal, create_db_and_tables
from models.models import User as UserModel # Renamed to avoid conflict with User schema if any
from auth_utils import verify_password # To check if password is not plain

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

    assert response.status_code == 422, response.text
    data = response.json()
    assert data["detail"] == "Password must be at least 8 characters long."

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
