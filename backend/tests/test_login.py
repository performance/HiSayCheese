import requests
import json

# --- Configuration ---
BASE_URL = "http://127.0.0.1:8000"
LOGIN_ENDPOINT = f"{BASE_URL}/api/auth/login"

# Your user credentials
login_payload = {
    'username': 'test@example.com',
    'password': 'StrongPassword1!'
}

def get_auth_token():
    """
    Makes a POST request to the /api/auth/login endpoint to get a token.
    """
    print(f"Attempting to log in user '{login_payload['username']}' at {LOGIN_ENDPOINT}")

    try:
        # The 'requests' library handles sending data as 'application/x-www-form-urlencoded'
        # correctly when you pass a dictionary to the 'data' parameter.
        response = requests.post(LOGIN_ENDPOINT, data=login_payload)

        # Raise an exception if the request returned an unsuccessful status code (4xx or 5xx)
        response.raise_for_status()

        # Parse the JSON response
        token_data = response.json()
        access_token = token_data.get('access_token')

        if access_token:
            print("\n✅ SUCCESS! Token received.")
            print(f"   Access Token: {access_token[:15]}...") # Print first 15 chars
            return access_token
        else:
            print("\n❌ FAILED! The response did not contain an access token.")
            print(f"   Status Code: {response.status_code}")
            print(f"   Response Body: {response.text}")
            return None

    except requests.exceptions.HTTPError as http_err:
        print(f"\n❌ FAILED! HTTP error occurred.")
        print(f"   Status Code: {http_err.response.status_code}")
        print(f"   Response Body: {http_err.response.text}")
        return None
    except requests.exceptions.RequestException as err:
        print(f"\n❌ FAILED! An ambiguous error occurred: {err}")
        return None

if __name__ == "__main__":
    # Make sure your FastAPI server is running before executing this script!
    get_auth_token()
