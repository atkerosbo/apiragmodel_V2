import faiss
from sqlalchemy.orm import Session
from models import EmbeddingsTable
from database import SessionLocal

INDEX_FILE = "index.faiss"
product_codes = []  # Global variable to store product codes
index = None        # Global FAISS index

def load_faiss_index(db: Session):
    """
    Load the FAISS index and product codes.
    """
    global index, product_codes
    print("Loading FAISS index...")
    try:
        # Load FAISS index
        index = faiss.read_index(INDEX_FILE)
        print("FAISS index loaded successfully.")

        # Load product codes
        product_codes.clear()
        product_codes.extend([row.code for row in db.query(EmbeddingsTable).all()])
        print(f"Loaded {len(product_codes)} product codes.")

    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        raise RuntimeError("Failed to load FAISS index.")

def get_faiss_resources():
    """
    Access the loaded FAISS index and product codes.
    """
    if index is None or not product_codes:
        raise RuntimeError("FAISS index is not loaded. Call load_faiss_index() during startup.")
    return index, product_codes
