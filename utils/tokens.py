from typing import  Annotated
from fastapi import Header, HTTPException
from dotenv import load_dotenv
import os

load_dotenv()

KEY = os.getenv("KEY")
TOKEN = os.getenv("TOKEN")

async def verify_token(x_token: Annotated[str, Header()]):
    if x_token != TOKEN:
        print(x_token)
        raise HTTPException(status_code=400, detail="X-Token header invalid")


async def verify_key(x_key: Annotated[str, Header()]):
    if x_key != KEY:
        print(x_key)
        raise HTTPException(status_code=400, detail="X-Key header invalid")
    return x_key
