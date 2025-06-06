# main.py
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse

from db.database import create_db_and_tables, get_db
from db import crud
from models import models

app = FastAPI()

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.get("/health", response_model=models.Number)
async def health(db: Session = Depends(get_db)):
    db_number = crud.get_number(db)
    if db_number is None:
        return JSONResponse(status_code=404, content={"message": "No number set yet"})
    return db_number

@app.post("/put_number", response_model=models.Number)
async def put_number(number: models.NumberCreate, db: Session = Depends(get_db)):
    return crud.create_or_update_number(db=db, value=number.value)

@app.post("/increment_number", response_model=models.Number)
async def increment_number_endpoint(db: Session = Depends(get_db)):
    db_number = crud.increment_number(db)
    if db_number is None:
        raise HTTPException(status_code=404, detail="No number set to increment.")
    return db_number

@app.get("/")
async def root():
    return {"message": "Hello World"}
