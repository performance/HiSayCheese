from sqlalchemy import create_engine, Column, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./sql_app.db" # Changed Database URL

Base = declarative_base()

class Number(Base):
    __tablename__ = "numbers"

    id = Column(Integer, primary_key=True, index=True)
    value = Column(Integer, index=True)

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} # Added connect_args
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) # Ensured SessionLocal is defined

def create_db_and_tables():
    Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
