# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# --- STEP 1: Import from your application ---
# We need the main app object to override its dependencies.
# We need Base from the models to create the tables.
# We need the get_db dependency function itself to use as a key for the override.
from main import app
from models.models import Base
from db.database import get_db


# --- STEP 2: Create a dedicated TEST database engine ---
# Use an in-memory SQLite database.
# The StaticPool is crucial for SQLite in-memory to work with multiple threads/sessions in tests.
# The connect_args is also required for SQLite.
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

# --- STEP 3: Create a dedicated TEST sessionmaker ---
# This will be used to create sessions for our tests.
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# --- STEP 4: The Core Fixture to set up and tear down the database ---
# This fixture will run for every single test function.
@pytest.fixture(scope="function")
def db_session():
    # Before the test runs, create all the tables in our in-memory database.
    # Base.metadata.create_all() is what your on_startup event does.
    Base.metadata.create_all(bind=engine)
    
    # Yield a new database session to the test function.
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        # After the test runs, close the session.
        db.close()
        # Drop all the tables, so the next test starts with a clean slate.
        Base.metadata.drop_all(bind=engine)


# --- STEP 5: The fixture that provides the configured TestClient ---
@pytest.fixture(scope="function")
def client(db_session):
    # This is the magic: override_get_db will be used instead of the real get_db
    # for the duration of this test.
    def override_get_db():
        try:
            yield db_session
        finally:
            db_session.close()

    # Apply the override to our app object.
    app.dependency_overrides[get_db] = override_get_db
    
    # Yield the configured TestClient.
    yield TestClient(app)

    # After the test, clear the override to be clean.
    app.dependency_overrides.clear()