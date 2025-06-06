from sqlalchemy.orm import Session
from .database import Number

def get_number(db: Session):
    return db.query(Number).first()

def create_or_update_number(db: Session, value: int):
    db_number = get_number(db)
    if db_number:
        db_number.value = value
    else:
        db_number = Number(value=value)
        db.add(db_number)
    db.commit()
    db.refresh(db_number)
    return db_number

def increment_number(db: Session):
    db_number = get_number(db)
    if db_number:
        db_number.value += 1
        db.commit()
        db.refresh(db_number)
        return db_number
    return None
