from pydantic import BaseModel

class NumberBase(BaseModel):
    value: int

class NumberCreate(NumberBase):
    pass

class Number(NumberBase):
    id: int

    class Config:
        orm_mode = True
