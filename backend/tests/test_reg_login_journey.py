import requests
import json

# --- Configuration ---
BASE_URL = "http://127.0.0.1:8000"
REGISTER_ENDPOINT = f"{BASE_URL}/api/auth/register"
LOGIN_ENDPOINT = f"{BASE_URL}/api/auth/login"

# Define our test user
test_user = {
    'email': 'test@example.com',
    'password': 'StrongPassword1!' # This password meets your Pydantic validator's criteria
}

# --- Helper Functions ---
def print_status(message, success=True):
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")

def handle_request_error(err):
    """A helper to print detailed error information from requests."""
    if isinstance(err, requests.exceptions.HTTPError):
        print_status(f"Request failed with status code {err.response.status_code}", success=False)
        try:
            # Try to print a formatted JSON error detail
            error_details = json.dumps(err.response.json(), indent=2)
            print(f"   Server Response:\n{error_details}")
        except json.JSONDecodeError:
            # Fallback for non-JSON responses
            print(f"   Server Response (raw):\n{err.response.text}")
    else:
        print_status(f"An unexpected request error occurred: {err}", success=False)

# --- Test Functions ---

def register_test_user():
    """Attempts to register the test user."""
    print("\n--- [Step 1: Registering User] ---")
    print(f"Attempting to register user '{test_user['email']}' at {REGISTER_ENDPOINT}")

    try:
        # For registration, the API expects a JSON body.
        response = requests.post(REGISTER_ENDPOINT, json=test_user)

        # Check for 409 Conflict, which means the user already exists (and is a success for our test)
        if response.status_code == 409:
            print_status(f"User '{test_user['email']}' already exists. That's OK!")
            return True

        response.raise_for_status() # Raise an error for other 4xx/5xx codes
        
        created_user = response.json()
        print_status(f"Successfully registered user with ID: {created_user.get('id')}")
        return True

    except requests.exceptions.RequestException as err:
        handle_request_error(err)
        return False

def login_test_user():
    """Attempts to log in the test user and returns the access token."""
    print("\n--- [Step 2: Logging In User] ---")
    print(f"Attempting to log in as '{test_user['email']}' at {LOGIN_ENDPOINT}")

    try:
        # For login, the API expects form data.
        login_payload = {
            'username': test_user['email'],
            'password': test_user['password']
        }
        response = requests.post(LOGIN_ENDPOINT, data=login_payload)
        response.raise_for_status()

        token_data = response.json()
        access_token = token_data.get('access_token')

        if access_token:
            print_status("Login successful. Token received.")
            # In a real test suite, you'd return this token.
            # For this script, we'll just print a snippet.
            print(f"   Access Token: {access_token[:20]}...")
            return access_token
        else:
            print_status("Login successful, but no token found in response.", success=False)
            return None

    except requests.exceptions.RequestException as err:
        handle_request_error(err)
        return None

def run_user_journey():
    """Executes the full user journey: register then log in."""
    # Step 1: Register the user. If this fails for any reason other
    # than the user already existing, we stop.
    if not register_test_user():
        print("\nHalting test journey due to registration failure.")
        return

    # Step 2: Log the user in.
    login_test_user()
    
    print("\n--- [User Journey Complete] ---")


if __name__ == "__main__":
    # Make sure your FastAPI server is running before executing this script!
    run_user_journey()
