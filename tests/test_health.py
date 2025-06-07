from fastapi.testclient import TestClient
from fastapi import status
from sqlalchemy.orm import Session # For type hinting mock
from unittest.mock import MagicMock

# Import the actual app instance from main.py
from main import app as main_app
# Import the actual get_db function used in the endpoint for overriding
from db.database import get_db as actual_get_db_for_override

client = TestClient(main_app)

def test_liveness_probe():
    """Test the /api/health/live endpoint returns 200 and correct body."""
    response = client.get("/api/health/live")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "alive"}

def test_readiness_probe_success():
    """
    Test the /api/health/ready endpoint returns 200 and correct body
    when the database connection is successful.
    This test relies on the actual database being accessible in the test environment.
    """
    original_overrides = main_app.dependency_overrides.copy()
    main_app.dependency_overrides.clear()

    response = client.get("/api/health/ready")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "ready", "detail": "Database connection successful."}

    main_app.dependency_overrides = original_overrides

def get_mock_db_that_fails():
    """
    A mock database session dependency that simulates a connection failure
    by raising an exception when db.execute is called.
    This must be a generator function as the actual get_db is a generator.
    """
    mock_session = MagicMock(spec=Session)
    mock_session.execute.side_effect = Exception("Simulated DB connection error")
    try:
        yield mock_session
    finally:
        pass

def test_readiness_probe_db_failure():
    """
    Test the /api/health/ready endpoint returns 503 and correct body
    when the database connection fails. Uses dependency override to simulate failure.
    """
    original_overrides = main_app.dependency_overrides.copy()

    main_app.dependency_overrides[actual_get_db_for_override] = get_mock_db_that_fails

    response = client.get("/api/health/ready")

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    # FastAPI's HTTPException nests the dictionary provided in `detail` under a "detail" key in the response.
    expected_json = {"detail": {"status": "database_error", "detail": "Cannot connect to database."}}
    assert response.json() == expected_json

    main_app.dependency_overrides = original_overrides
