# tests/test_image_flow.py
import pytest
import uuid
from unittest.mock import patch

# --- Module Imports ---
from models.models import User as UserModel
from .conftest import TestingSessionLocal # Used to verify DB state independently.


def test_full_user_and_image_flow(client):
    """
    Tests the full end-to-end user journey in a single, integrated test:
    1. Register a new user.
    2. Verify the user's email by fetching the token from the DB.
    3. Log in as the verified user to get an auth token.
    4. Upload an image using the auth token.
    5. Apply an enhancement to the uploaded image.
    """
    # --- [Step 1: Registration] ---
    user_email = f"full_flow_{uuid.uuid4()}@example.com"
    password = "vAl!dpassword123"

    # We patch the email service as its functionality is tested elsewhere.
    with patch("services.email_service.EmailService.send_email"):
        reg_response = client.post(
            "/api/auth/register",
            json={"email": user_email, "password": password}
        )
    assert reg_response.status_code == 201, "Step 1 Failed: Registration"

    # --- [Step 2: Email Verification] ---
    # In a test, we "cheat" by getting the token directly from the database.
    db = TestingSessionLocal()
    try:
        user_in_db = db.query(UserModel).filter(UserModel.email == user_email).first()
        assert user_in_db is not None, "User not found in DB after registration."
        verification_token = user_in_db.verification_token
    finally:
        db.close()
    
    assert verification_token is not None, "Verification token was not created."

    # "Click" the verification link.
    verify_response = client.get(f"/api/auth/verify-email?token={verification_token}")
    assert verify_response.status_code == 200, "Step 2 Failed: Email Verification"
    assert verify_response.json()["is_verified"] is True

    # --- [Step 3: Login] ---
    login_response = client.post(
        "/api/auth/login",
        data={"username": user_email, "password": password}
    )
    assert login_response.status_code == 200, "Step 3 Failed: Login"
    access_token = login_response.json()["access_token"]
    auth_headers = {"Authorization": f"Bearer {access_token}"}

    # --- [Step 4: Image Upload] ---
    # Create a dummy image file content in memory
    from io import BytesIO
    from PIL import Image
    
    img_byte_arr = BytesIO()
    image = Image.new('RGB', (100, 100), 'red')
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0) # Rewind the buffer to the beginning

    files = {'file': ('test_flow_image.png', img_byte_arr, 'image/png')}
    upload_response = client.post("/api/images/upload", headers=auth_headers, files=files)
    
    assert upload_response.status_code == 201, f"Step 4 Failed: Image Upload. Response: {upload_response.text}"
    image_id = upload_response.json()["id"]

    # --- [Step 5: Apply Enhancements] ---
    enhancement_payload = {
        "image_id": image_id,
        "parameters": {
            "brightness_target": 1.1,
            "contrast_target": 1.05,
            "saturation_target": 1.0,
            "background_blur_radius": 0,
            "crop_rect": [0, 0, 100, 100],
            "face_smooth_intensity": 0
        }
    }
    
    enhancement_response = client.post(
        "/api/enhancement/apply",
        headers=auth_headers,
        json=enhancement_payload
    )

    assert enhancement_response.status_code == 200, f"Step 5 Failed: Apply Enhancements. Response: {enhancement_response.text}"
    enhancement_data = enhancement_response.json()
    assert enhancement_data["original_image_id"] == image_id
    assert "processed_image_id" in enhancement_data
    assert "processed_image_path" in enhancement_data
    
    print("\n✅ Full end-to-end image flow test PASSED!")