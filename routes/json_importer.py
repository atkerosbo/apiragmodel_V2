import json
from sentence_transformers import SentenceTransformer
import numpy as np
from fastapi import Depends, APIRouter, UploadFile, File, HTTPException
from utils.tokens import verify_key, verify_token
import os
import shutil
from utils.embedding_processor import main_embedding_process
from models import Data, Products, EmbeddingsTable, Information, InformationEmbeddings

from utils.embeding_to_database import save_products_to_database, save_desription_to_database
from sqlalchemy.orm import Session
from database import get_db
from dotenv import load_dotenv
import logging, traceback

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploads"
router = APIRouter()

# Initialize embedding model
embedder = SentenceTransformer("all-mpnet-base-v2")


@router.post("/json-upload-products/", dependencies=[Depends(verify_token), Depends(verify_key)], status_code=200)
async def upload_json(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Validate file type
    if not file.filename.lower().endswith('.json'):
        raise HTTPException(status_code=400, detail="File must be a JSON")
    
    # Create full path for saving the file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        # Ensure upload directory exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Import JSON to database
        save_products_to_database(file_path, db)


        # make embeddings
        main_embedding_process(db, Products, EmbeddingsTable)


        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with decoding and embedding: {str(e)}")
    
    return {"filename": file.filename, "message": "JSON uploaded, processed, and saved successfully"}


@router.post("/json-upload-description/", dependencies=[Depends(verify_token), Depends(verify_key)], status_code=200)
async def upload_json(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Validate file type
    if not file.filename.lower().endswith('.json'):
        raise HTTPException(status_code=400, detail="File must be a JSON")
    
    # Create full path for saving the file
    json_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        # Ensure upload directory exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # Save the uploaded file
        with open(json_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(json_path)
        # Import JSON to database
        save_desription_to_database(json_path, db)


         # make embeddings
        main_embedding_process(db, Information, InformationEmbeddings)


        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with decoding and embedding: {str(e)}")
    
    return {"filename": file.filename, "message": "JSON uploaded, processed, and saved successfully"}