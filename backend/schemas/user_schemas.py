# schemas/user_schemas.py
import re
import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, constr, validator

MIN_PASSWORD_LENGTH = 8

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: constr(min_length=MIN_PASSWORD_LENGTH)

    @validator('password')
    def password_strength(cls, v):
        if not re.search(r"[A-Z]", v):
            raise ValueError('Password must contain an uppercase letter')
        if not re.search(r"[a-z]", v):
            raise ValueError('Password must contain a lowercase letter')
        if not re.search(r"\d", v):
            raise ValueError('Password must contain a digit')
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError('Password must contain a special character')
        return v

class UserSchema(UserBase):
    id: uuid.UUID
    created_at: datetime
    is_verified: bool

    class Config:
        from_attributes = True