from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.models import Base, Number, Image, User, EnhancementHistory # Import Base and models

DATABASE_URL = "sqlite:///./sql_app.db" # Changed Database URL

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} # Added connect_args
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) # Ensured SessionLocal is defined

def create_db_and_tables():
    Base.metadata.create_all(bind=engine) # This will now create all tables defined in models.models

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
