import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from main import app
from db.database import Base, get_db, Number # Ensure Number is imported

# Setup for in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Override get_db dependency for testing
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture()
def client():
    Base.metadata.create_all(bind=engine)  # Create tables
    yield TestClient(app)
    Base.metadata.drop_all(bind=engine)    # Drop tables after tests

# Test functions
def test_health_no_number(client):
    response = client.get("/health")
    assert response.status_code == 404
    assert response.json() == {"message": "No number set yet"}

def test_put_number(client):
    response = client.post("/put_number", json={"value": 10})
    assert response.status_code == 200
    # Assuming the first item will have id=1 in a clean in-memory db
    assert response.json() == {"id": 1, "value": 10}

def test_health_get_number(client):
    client.post("/put_number", json={"value": 20}) # Set a number first
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"id": 1, "value": 20}

def test_increment_number_no_initial(client):
    response = client.post("/increment_number")
    assert response.status_code == 404
    assert response.json()["detail"] == "No number set to increment."

def test_increment_number(client):
    client.post("/put_number", json={"value": 5}) # Set initial number
    response = client.post("/increment_number")
    assert response.status_code == 200
    assert response.json() == {"id": 1, "value": 6}

def test_put_increment_get_sequence(client):
    response_put = client.post("/put_number", json={"value": 100})
    assert response_put.status_code == 200
    assert response_put.json() == {"id": 1, "value": 100}

    response_inc1 = client.post("/increment_number")
    assert response_inc1.status_code == 200
    assert response_inc1.json() == {"id": 1, "value": 101}

    response_inc2 = client.post("/increment_number")
    assert response_inc2.status_code == 200
    assert response_inc2.json() == {"id": 1, "value": 102}

    response_health = client.get("/health")
    assert response_health.status_code == 200
    assert response_health.json() == {"id": 1, "value": 102}
