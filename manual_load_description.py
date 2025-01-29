import faiss
import numpy as np
from database import SessionLocal
from models import EmbeddingsTable, InformationEmbeddings , SeparateEmbeddingTables

DESCRIPTION_INDEX_FILE = "separate_index.faiss"


def create_faiss_index_description():
    db = SessionLocal()
    try:
        print("Fetching embeddings from the InformationEmbeddings table...")
        rows = db.query(SeparateEmbeddingTables).all()
        print(f"Number of rows fetched: {len(rows)}")

        embeddings = []
        information_codes = []
        for row in rows:
            try:
                embedding = np.frombuffer(row.embedding, dtype=np.float32)
                embeddings.append(embedding)
                information_codes.append(row.code)
            except Exception as e:
                print(f"Error processing row {row.id}: {e}")

        if not embeddings:
            print("No embeddings found in the InformationEmbeddings table.")
            return

        # Convert embeddings to a numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        print(f"Loaded {len(embeddings_array)} embeddings.")

        # Build FAISS index
        dimension = embeddings_array.shape[1]  # Embedding dimension
        separate_index = faiss.IndexFlatL2(dimension)  # L2 distance index
        separate_index.add(embeddings_array)

        # Save the index
        faiss.write_index(separate_index, DESCRIPTION_INDEX_FILE)
        print(f"FAISS index saved to {DESCRIPTION_INDEX_FILE}.")

    except Exception as e:
        print(f"Error creating FAISS description index: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    create_faiss_index_description()