# config.py

# IMPORTANT: This is a default secret key for development purposes ONLY.
# For production, use a strong, randomly generated key and load it from
# an environment variable or a secure configuration management system.
SECRET_KEY: str = "your-super-secret-key-please-change-in-production"

ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
