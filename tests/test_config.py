import os
import pytest
import importlib

# Define a dictionary of non-placeholder values for testing successful config loading
MINIMAL_REQUIRED_CONFIG_FOR_TEST_NON_PLACEHOLDER = {
    "SECRET_KEY": "a_real_secret_key_for_testing",
    "ALGORITHM": "HS256",  # Has a hardcoded default, not a placeholder to check against
    "ACCESS_TOKEN_EXPIRE_MINUTES": "30",  # Also hardcoded default
    "AWS_S3_BUCKET_NAME": "my-actual-test-bucket",
    "AWS_S3_REGION": "us-west-2", # Default in config.py is us-east-1, so this is a change
    "AWS_ACCESS_KEY_ID": "ACTUAL_TEST_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY": "ACTUAL_TEST_SECRET_ACCESS_KEY",
    "AWS_SES_REGION": "us-west-2", # Default in config.py is S3 region, then us-east-1
    "AWS_SES_SENDER_EMAIL": "real-sender-for-test@example.com",
    "FRONTEND_URL": "https://test.app.com", # Default is http://localhost:3000
    "SENTRY_DSN": "https://test-sentry-dsn@sentry.io/12345", # Default is placeholder but acceptable
    "DATABASE_URL": "sqlite:///./test_suite_app.db" # Default is sqlite:///./sql_app.db
}

@pytest.fixture(autouse=True)
def manage_environment():
    """
    Manages os.environ for each test.
    - Stores the original environment.
    - Clears predefined config keys before the test.
    - Restores the original environment after the test.
    """
    original_env = os.environ.copy()

    # Keys that config.py reads via os.getenv()
    # This list should cover all variables defined at the global scope in config.py
    # that are populated using os.getenv()
    config_keys_to_clear = [
        "SECRET_KEY", "ALGORITHM", "ACCESS_TOKEN_EXPIRE_MINUTES",
        "AWS_S3_BUCKET_NAME", "AWS_S3_REGION", "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY", "AWS_SES_REGION", "AWS_SES_SENDER_EMAIL",
        "FRONTEND_URL", "SENTRY_DSN", "DATABASE_URL"
    ]

    for key in config_keys_to_clear:
        if key in os.environ:
            del os.environ[key]

    yield # Test runs here

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

def test_config_loading_success():
    """
    Tests that config.py loads successfully and validate_configuration() passes
    when all required environment variables are set to non-placeholder values.
    """
    for key, value in MINIMAL_REQUIRED_CONFIG_FOR_TEST_NON_PLACEHOLDER.items():
        os.environ[key] = value

    try:
        import config
        importlib.reload(config) # Reload to trigger definitions and validation
        # validate_configuration is called at the end of config.py
    except ValueError as e:
        pytest.fail(f"Configuration validation failed unexpectedly: {e}")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred during config loading: {e}")

def test_config_loading_failure_placeholder_secret_key():
    """
    Tests that validate_configuration() fails if SECRET_KEY is not set in the environment,
    causing it to fall back to its placeholder default value in config.py.
    """
    # Set all other variables to non-placeholder values
    for key, value in MINIMAL_REQUIRED_CONFIG_FOR_TEST_NON_PLACEHOLDER.items():
        if key != "SECRET_KEY":
            os.environ[key] = value

    # SECRET_KEY is deliberately not set in os.environ for this test.
    # config.py will assign its placeholder default: "your-super-secret-key-please-change-in-production"

    with pytest.raises(ValueError, match=r"SECRET_KEY .*is set to a default placeholder value.*and must be changed"):
        import config
        importlib.reload(config) # Reload to trigger definitions and validation

def test_config_loading_failure_placeholder_aws_s3_bucket_name():
    """
    Tests that validate_configuration() fails if AWS_S3_BUCKET_NAME uses its placeholder.
    """
    for key, value in MINIMAL_REQUIRED_CONFIG_FOR_TEST_NON_PLACEHOLDER.items():
        if key != "AWS_S3_BUCKET_NAME":
            os.environ[key] = value

    # AWS_S3_BUCKET_NAME will use its placeholder "your-s3-bucket-name"
    with pytest.raises(ValueError, match=r"AWS_S3_BUCKET_NAME .*is set to a default placeholder value.*and must be changed"):
        import config
        importlib.reload(config)

def test_config_loading_failure_placeholder_aws_access_key_id():
    """
    Tests that validate_configuration() fails if AWS_ACCESS_KEY_ID uses its placeholder.
    """
    for key, value in MINIMAL_REQUIRED_CONFIG_FOR_TEST_NON_PLACEHOLDER.items():
        if key != "AWS_ACCESS_KEY_ID":
            os.environ[key] = value

    with pytest.raises(ValueError, match=r"AWS_ACCESS_KEY_ID .*is set to a default placeholder value.*and must be changed"):
        import config
        importlib.reload(config)

def test_config_loading_failure_placeholder_aws_secret_access_key():
    """
    Tests that validate_configuration() fails if AWS_SECRET_ACCESS_KEY uses its placeholder.
    """
    for key, value in MINIMAL_REQUIRED_CONFIG_FOR_TEST_NON_PLACEHOLDER.items():
        if key != "AWS_SECRET_ACCESS_KEY":
            os.environ[key] = value

    with pytest.raises(ValueError, match=r"AWS_SECRET_ACCESS_KEY .*is set to a default placeholder value.*and must be changed"):
        import config
        importlib.reload(config)

def test_config_loading_failure_placeholder_aws_ses_sender_email():
    """
    Tests that validate_configuration() fails if AWS_SES_SENDER_EMAIL uses its placeholder.
    """
    for key, value in MINIMAL_REQUIRED_CONFIG_FOR_TEST_NON_PLACEHOLDER.items():
        if key != "AWS_SES_SENDER_EMAIL":
            os.environ[key] = value

    with pytest.raises(ValueError, match=r"AWS_SES_SENDER_EMAIL .*is set to a default placeholder value.*and must be changed"):
        import config
        importlib.reload(config)

# Example of a test for a variable that should NOT cause validation failure if it's a placeholder
# This depends on SENTRY_DSN not being in critical_vars_and_placeholders in config.py
def test_sentry_dsn_placeholder_is_acceptable():
    """
    Tests that if SENTRY_DSN is its placeholder, validation still passes,
    assuming this placeholder is considered acceptable (e.g., Sentry is optional).
    """
    for key, value in MINIMAL_REQUIRED_CONFIG_FOR_TEST_NON_PLACEHOLDER.items():
        if key != "SENTRY_DSN": # Skip setting SENTRY_DSN, so it uses placeholder
            os.environ[key] = value

    # SENTRY_DSN is not set in os.environ, so config.py will use its placeholder.
    # We expect this to NOT raise an error from validate_configuration.
    try:
        import config
        importlib.reload(config)
    except ValueError as e:
        if "SENTRY_DSN" in str(e): # Make sure it's not failing due to Sentry DSN
            pytest.fail(f"SENTRY_DSN placeholder caused unexpected validation failure: {e}")
        # If it's another error, let it propagate or fail generally
        # pytest.fail(f"Configuration validation failed unexpectedly: {e}")
        # For this test, we only care if SENTRY_DSN specifically caused an issue.
        # If other issues arise, other tests should catch them.
        # A more robust way is to ensure SENTRY_DSN is NOT in critical_vars_and_placeholders
        # and then this test just runs like test_config_loading_success but without SENTRY_DSN set.
    # If no ValueError or if ValueError is not about SENTRY_DSN, it's a pass for this specific check.
    # This test is a bit tricky; the main success test already covers Sentry DSN being non-placeholder.
    # This one checks that the *placeholder* for Sentry DSN is okay.
    # The easiest way to ensure this is to make sure SENTRY_DSN is not in critical_vars_and_placeholders
    # in config.py, which it currently is not.

    # Reset SENTRY_DSN to a non-placeholder to be sure it doesn't affect other tests if manage_environment isn't perfect.
    os.environ["SENTRY_DSN"] = MINIMAL_REQUIRED_CONFIG_FOR_TEST_NON_PLACEHOLDER["SENTRY_DSN"]
    importlib.reload(config) # Re-reload with a good Sentry DSN

# Test that ALGORITHM and ACCESS_TOKEN_EXPIRE_MINUTES use hardcoded if not in env
def test_hardcoded_defaults_used_if_not_in_env():
    """
    Tests that ALGORITHM and ACCESS_TOKEN_EXPIRE_MINUTES use their hardcoded Python
    defaults if not present in the environment, and this does not cause validation failure.
    """
    # Set all required, but specifically exclude ALGORITHM and ACCESS_TOKEN_EXPIRE_MINUTES from os.environ
    for key, value in MINIMAL_REQUIRED_CONFIG_FOR_TEST_NON_PLACEHOLDER.items():
        if key not in ["ALGORITHM", "ACCESS_TOKEN_EXPIRE_MINUTES"]:
            os.environ[key] = value

    # ALGORITHM and ACCESS_TOKEN_EXPIRE_MINUTES are not in os.environ.
    # config.py should use their hardcoded values: "HS256" and 30.
    # validate_configuration should not complain about these.
    try:
        import config
        importlib.reload(config)
        assert config.ALGORITHM == "HS256" # Check they have their Python-defined defaults
        assert config.ACCESS_TOKEN_EXPIRE_MINUTES == 30
    except ValueError as e:
        pytest.fail(f"Validation failed with ALGORITHM/ACCESS_TOKEN_EXPIRE_MINUTES not in env: {e}")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred: {e}")

    # Restore them for subsequent tests if manage_environment isn't fully isolating (it should be)
    os.environ["ALGORITHM"] = MINIMAL_REQUIRED_CONFIG_FOR_TEST_NON_PLACEHOLDER["ALGORITHM"]
    os.environ["ACCESS_TOKEN_EXPIRE_MINUTES"] = MINIMAL_REQUIRED_CONFIG_FOR_TEST_NON_PLACEHOLDER["ACCESS_TOKEN_EXPIRE_MINUTES"]
    importlib.reload(config)
