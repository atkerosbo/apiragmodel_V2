import faiss
from sqlalchemy.orm import Session
from models import SeparateEmbeddingTables
from database import SessionLocal

DESCRIPTION_INDEX_FILE = "separate_index.faiss"
information_codes = []  # Global variable to store product codes
separate_index = None        # Global FAISS index

def load_separate_faiss_index(db: Session):
    """
    Load the FAISS index and information codes.
    """
    global separate_index, information_codes
    print("Loading FAISS index...")
    try:
        # Load FAISS index
        separate_index = faiss.read_index(DESCRIPTION_INDEX_FILE)
        print("FAISS description index loaded successfully.")

        # Load product codes
        information_codes.clear()
        information_codes.extend([row.code for row in db.query(SeparateEmbeddingTables).all()])
        print(f"Loaded {len(information_codes)} information codes.")

    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        raise RuntimeError("Failed to load FAISS index.")

def get_separate_faiss_index_resources():
    """
    Access the loaded FAISS index and description codes.
    """
    if separate_index is None or not information_codes:
        raise RuntimeError("FAISS index is not loaded. Call load_faiss_index() during startup.")
    return separate_index, information_codes
