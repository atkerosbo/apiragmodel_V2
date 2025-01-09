from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
import os
from database import engine, SessionLocal
import models
from routes import semantic_search, rag_query_endpoint, json_importer
from faiss_index import load_faiss_index
from description_faiss_index import load_description_faiss_index
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

models.Base.metadata.create_all(bind=engine)


# Globals for FAISS index and product codes
INDEX_FILE = "index.faiss"
product_codes = []
index = None

# Define allowed IPs
ALLOWED_IPS = ["127.0.0.1", "144.76.67.177", "109.92.201.78"]  # Replace with the IPs you want to allow


@asynccontextmanager
async def lifespan(app):
    print("Application startup: Loading FAISS index...")
    db = SessionLocal()  # Create a new session
    try:
        load_faiss_index(db)  # Load FAISS index and product codes
        print("FAISS index loaded successfully.")
        load_description_faiss_index(db)  # Load FAISS Description index and information codes
        print("Description FAISS index loaded successfully.")
        
        yield
    finally:
        db.close()
        print("Database connection closed.")

app = FastAPI(lifespan=lifespan)

# Middleware to restrict IP access
@app.middleware("http")
async def restrict_ip_middleware(request: Request, call_next):
    client_ip = request.client.host
    if client_ip not in ALLOWED_IPS:
        raise HTTPException(status_code=403, detail="Access forbidden: Your IP is not allowed.")
    response = await call_next(request)
    return response


origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers

app.include_router(semantic_search.router)
app.include_router(rag_query_endpoint.router)
app.include_router(json_importer.router)



@app.get("/")
async def root():
    return {"message": "Hello World"}



# atexit.register(clean_up_semaphores)





