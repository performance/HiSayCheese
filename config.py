import os

# config.py

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
