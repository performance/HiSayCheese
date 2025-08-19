import logging # Added
from sqlalchemy import create_engine
# from sqlalchemy.ext.declarative import declarative_base # Now imported from models.models
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL
from models.models import Base # Import Base from models.models

logger = logging.getLogger(__name__) # Added
logger.info("db.database module loaded.") # LOGGING

# Print type and value of DATABASE_URL to see what it is at import time
logger.info(f"DATABASE_URL at import time: {DATABASE_URL} (type: {type(DATABASE_URL)})")


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}) # Added connect_args for SQLite
logger.info(f"Default engine created: {engine} for DATABASE_URL: {DATABASE_URL}") # LOGGING

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
logger.info(f"Default SessionLocal created, bound to engine: {engine}") # LOGGING

# Base = declarative_base() # Base is now imported from models.models

def get_db():
    # This get_db will be patched by conftest.py during tests
    # to use a test-specific SessionLocal.
    db = SessionLocal()
    logger.debug(f"get_db called, returning session: {db} (from SessionLocal: {SessionLocal})")
    try:
        yield db
    finally:
        logger.debug(f"Closing session: {db}")
        db.close()

def create_db_and_tables():
    # This function will use whatever 'engine' is currently set in this module's scope.
    # In tests, conftest.py patches this module's 'engine' to be the test_engine.
    logger.info(f"create_db_and_tables called. Current engine: {engine}") # LOGGING
    try:
        Base.metadata.create_all(bind=engine)
        logger.info(f"Base.metadata.create_all(bind=engine) completed using engine: {engine}.") # LOGGING
    except Exception as e:
        logger.error(f"Error during Base.metadata.create_all with engine {engine}: {e}", exc_info=True)
        raise
