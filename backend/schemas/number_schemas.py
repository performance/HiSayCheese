# schemas/number_schemas.py
from pydantic import BaseModel, conint

class NumberBase(BaseModel):
    value: conint(ge=0)

class NumberCreate(NumberBase):
    pass

class NumberSchema(NumberBase):
    id: int

    class Config:
        from_attributes = True