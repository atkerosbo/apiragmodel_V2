from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv
import os

load_dotenv()

DBUSER = os.getenv("DBUSER")
DBPASSWORD = os.getenv("DBPASSWORD")
DBNAME = os.getenv("DBNAME")


SQLALCHEMY_DATABASE_URL = f"postgresql://{DBUSER}:{DBPASSWORD}@localhost:5432/{DBNAME}"

# SQLALCHEMY_DATABASE_URL = f"postgresql://user:Komsrv999@128.140.125.68:5432/mydb"

engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()