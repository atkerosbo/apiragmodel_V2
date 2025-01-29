import faiss
import numpy as np
from database import SessionLocal
from models import EmbeddingsTable, InformationEmbeddings, SeparateEmbeddingTables
INDEX_FILE = "index.faiss"
DESCRIPTION_INDEX_FILE = "description_index.faiss"
SEPARATE_INDEX_FILE = "separate_index.faiss"

def create_faiss_index():
    db = SessionLocal()
    try:
        print("Fetching embeddings from the database...")
        rows = db.query(EmbeddingsTable).all()

        embeddings = []
        product_codes = []
        for row in rows:
            embedding = np.frombuffer(row.embedding, dtype=np.float32)
            embeddings.append(embedding)
            product_codes.append(row.code)

        if not embeddings:
            print("No embeddings found in the database.")
            return

        # Convert embeddings to a numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        print(f"Loaded {len(embeddings_array)} embeddings.")

        # Build FAISS index
        dimension = embeddings_array.shape[1]  # Embedding dimension
        index = faiss.IndexFlatL2(dimension)  # L2 distance index
        index.add(embeddings_array)

        # Save the index
        faiss.write_index(index, INDEX_FILE)
        print(f"FAISS index saved to {INDEX_FILE}.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()


def create_faiss_index_description():
    db = SessionLocal()
    try:
        print("Fetching embeddings from the database...")
        rows = db.query(InformationEmbeddings).all()

        embeddings = []
        information_codes = []
        for row in rows:
            embedding = np.frombuffer(row.embedding, dtype=np.float32)
            embeddings.append(embedding)
            information_codes.append(row.code)

        if not embeddings:
            print("No embeddings found in the database.")
            return

        # Convert embeddings to a numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        print(f"Loaded {len(embeddings_array)} embeddings.")

        # Build FAISS index
        dimension = embeddings_array.shape[1]  # Embedding dimension
        description_index = faiss.IndexFlatL2(dimension)  # L2 distance index
        description_index.add(embeddings_array)

        # Save the index
        faiss.write_index(description_index, DESCRIPTION_INDEX_FILE)
        print(f"FAISS index saved to {DESCRIPTION_INDEX_FILE}.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()       

def create_faiss_index_separate():
    db = SessionLocal()
    try:
        print("Fetching embeddings from the database...")
        rows = db.query(SeparateEmbeddingTables).all()

        embeddings = []
        information_codes = []
        for row in rows:
            embedding = np.frombuffer(row.embedding, dtype=np.float32)
            embeddings.append(embedding)
            information_codes.append(row.code)

        if not embeddings:
            print("No embeddings found in the database.")
            return

        # Convert embeddings to a numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        print(f"Loaded {len(embeddings_array)} embeddings.")

        # Build FAISS index
        dimension = embeddings_array.shape[1]  # Embedding dimension
        separate_index = faiss.IndexFlatL2(dimension)  # L2 distance index
        separate_index.add(embeddings_array)

        # Save the index
        faiss.write_index(separate_index, SEPARATE_INDEX_FILE)
        print(f"FAISS index saved to {SEPARATE_INDEX_FILE}.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()       


if __name__ == "__main__":
    create_faiss_index()
    create_faiss_index_description()
    create_faiss_index_separate()
