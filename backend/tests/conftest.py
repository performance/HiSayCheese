import pytest
import os
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from unittest.mock import patch, MagicMock

# Ensure other project modules can be imported if necessary
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app  # Adjust if your FastAPI app instance is named differently or located elsewhere
from db.database import Base, get_db # Adjust import path as per your project structure
from config import Settings, get_settings # For overriding settings if needed

# 1. Pytest-Asyncio Configuration (though often it works by just being installed)
#    No explicit configuration needed in conftest.py for basic usage.
#    Ensure tests using async features are marked with @pytest.mark.asyncio

# 2. In-memory SQLite Database Fixture
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency override for get_db
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    # Create tables at the start of the test session
    Base.metadata.create_all(bind=engine)
    yield
    # Drop tables at the end of the test session (optional, as it's in-memory)
    # Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function") # function scope ensures DB is clean for each test
def db_session(setup_test_database): # Depends on session-scoped setup
    # Apply the override for the duration of tests using this fixture
    app.dependency_overrides[get_db] = override_get_db
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.rollback() # Rollback any changes to ensure test isolation
        db.close()
    # Clear overrides after test, if not managed by fixture scope
    app.dependency_overrides.clear()


# 3. TestClient Fixture
@pytest.fixture(scope="module")
def client(setup_test_database): # Depends on session-scoped setup
    # Override get_db for the TestClient's lifespan within a module
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    # Clear overrides after client usage if necessary
    app.dependency_overrides.clear()

# 4. Mock External Services (S3 and SES)
@pytest.fixture(scope="module", autouse=True) # Autouse for module scope
def mock_boto3_module():
    # This mock will apply to all tests in a module.
    # For finer control, apply directly in tests or specific fixtures.
    # Mock S3 client
    mock_s3_client = MagicMock()
    mock_s3_client.upload_fileobj.return_value = None
    mock_s3_client.download_fileobj.return_value = None
    mock_s3_client.delete_object.return_value = None
    mock_s3_client.generate_presigned_url.return_value = "http://mocked.s3.url/test-file"
    mock_s3_client.head_bucket.return_value = True # To simulate bucket exists

    # Mock SES client
    mock_ses_client = MagicMock()
    mock_ses_client.send_email.return_value = {"MessageId": "mock_message_id"}

    # Patch boto3.client to return these mocks
    # Adjust the actual path to 'boto3.client' based on where it's called in your services
    # e.g., 'services.storage_service.boto3.client' or 'services.email_service.boto3.client'
    # For a general approach, you might need to patch where StorageService/EmailService get their clients.

    # It's often better to mock the service classes themselves or their client instances.
    # Example: patching the client attribute if it's set like self.s3_client = boto3.client('s3')
    # For now, this is a placeholder. More specific patching might be needed.
    # If StorageService and EmailService instantiate clients directly, e.g., in __init__,
    # then patching 'boto3.client' globally can work if done early enough.

    # Let's assume services might import boto3 directly:
    s3_patch = patch('boto3.client', side_effect=lambda service_name, **kwargs: mock_s3_client if service_name == 's3' else mock_ses_client if service_name == 'ses' else MagicMock())

    # More robust: Patch at the point of use in your service classes
    # This is highly dependent on your StorageService and EmailService implementation
    # Example:
    # storage_service_s3_client_patch = patch('backend.services.storage_service.StorageService.s3_client', new_callable=PropertyMock(return_value=mock_s3_client))
    # email_service_ses_client_patch = patch('backend.services.email_service.EmailService.ses_client', new_callable=PropertyMock(return_value=mock_ses_client))


    # For now, we'll use a general boto3.client patch.
    # This might need adjustment if your services import/use boto3 differently.
    # A common pattern is `from boto3 import client as boto3_client` then `boto3_client('s3')`
    # Or `import boto3` then `boto3.client('s3')`

    # Patching 'boto3.client' globally
    boto3_client_patch = patch('boto3.client')
    mock_boto_client = boto3_client_patch.start()

    def side_effect_func(service_name, *args, **kwargs):
        if service_name == 's3':
            return mock_s3_client
        elif service_name == 'ses':
            return mock_ses_client
        # Fallback for other services if any, or raise an error
        raise ValueError(f"Unexpected service_name for boto3.client mock: {service_name}")

    mock_boto_client.side_effect = side_effect_func

    yield # Mocks are active during this yield

    boto3_client_patch.stop()
    # storage_service_s3_client_patch.stop()
    # email_service_ses_client_patch.stop()


# Fixture to override settings, e.g., for test-specific API keys or features
@pytest.fixture(scope="function")
def test_settings(monkeypatch):
    # Example: Override a specific setting
    # monkeypatch.setattr("config.Settings.some_setting", "test_value")
    # return get_settings() # Return the modified settings instance
    # For now, this is a placeholder.
    pass

# Example of how to use db_session in a test:
# def test_example(db_session: Session):
#     # db_session is the SQLAlchemy session for your test DB
#     user = db_session.query(UserModel).first() # Example query
#     assert user is None

# Example of how to use client in a test:
# def test_api_example(client):
#     response = client.get("/some/api/endpoint")
#     assert response.status_code == 200
