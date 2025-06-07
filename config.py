import os
from dotenv import load_dotenv

# config.py

load_dotenv()

# IMPORTANT: This is a default secret key for development purposes ONLY.
# For production, use a strong, randomly generated key and load it from
# an environment variable or a secure configuration management system.
SECRET_KEY: str = os.getenv("SECRET_KEY", "your-super-secret-key-please-change-in-production")

ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

# AWS S3 Configuration
AWS_S3_BUCKET_NAME: str = os.getenv("AWS_S3_BUCKET_NAME", "your-s3-bucket-name")
AWS_S3_REGION: str = os.getenv("AWS_S3_REGION", "us-east-1")
AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "your-access-key-id")
AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "your-secret-access-key")

# AWS SES Configuration (reuses S3 access keys if IAM permissions allow)
AWS_SES_REGION: str = os.getenv("AWS_SES_REGION", AWS_S3_REGION) # Default to S3 region
AWS_SES_SENDER_EMAIL: str = os.getenv("AWS_SES_SENDER_EMAIL", "sender@example.com") # Must be verified in SES

# Frontend URL (for email links)
FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Sentry Configuration
SENTRY_DSN: str = os.getenv("SENTRY_DSN", "your-sentry-dsn-goes-here") # Placeholder DSN

# Database URL - already present in .env.example, ensure it's loaded
DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./sql_app.db")


def validate_configuration():
    """
    Validates that critical configuration variables are not set to their
    default placeholder values.
    Raises ValueError if any critical variable is a placeholder.
    """
    critical_vars_and_placeholders = {
        "SECRET_KEY": "your-super-secret-key-please-change-in-production",
        "AWS_S3_BUCKET_NAME": "your-s3-bucket-name",
        "AWS_ACCESS_KEY_ID": "your-access-key-id",
        "AWS_SECRET_ACCESS_KEY": "your-secret-access-key",
        "AWS_SES_SENDER_EMAIL": "sender@example.com",
        # SENTRY_DSN's placeholder "your-sentry-dsn-goes-here" is considered acceptable
        # as it implies Sentry is intentionally not configured.
        # FRONTEND_URL's default "http://localhost:3000" is acceptable for local dev.
        # ALGORITHM and ACCESS_TOKEN_EXPIRE_MINUTES have hardcoded defaults, not placeholders.
        # DATABASE_URL's default "sqlite:///./sql_app.db" is acceptable for local dev.
    }
    problematic_vars = []
    for var_name, placeholder in critical_vars_and_placeholders.items():
        current_value = globals().get(var_name)
        if current_value == placeholder:
            problematic_vars.append(
                f"{var_name} (is set to a default placeholder value: '{placeholder}' and must be changed)"
            )
        elif current_value is None and os.getenv(var_name) is None:
            # This case handles if the variable itself is somehow not defined in globals()
            # or if os.getenv(var_name) without a default would return None (should not happen with current config.py structure)
            problematic_vars.append(f"{var_name} (is missing or not loaded correctly)")

    if problematic_vars:
        raise ValueError(
            "Configuration problems found:\n - " + "\n - ".join(problematic_vars)
        )

validate_configuration()
