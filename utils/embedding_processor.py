import numpy as np
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sqlalchemy.exc import SQLAlchemyError

from tqdm import tqdm  # To monitor batch progress
from math import ceil

import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")



# Initialize tokenizer and model constants
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MAX_LENGTH = 512  # Max token length for truncation

class EmbeddingProcessor:
    def __init__(self, db: Session, embedding_model):
        self.db = db
        self.embedding_model = embedding_model

    def fetch_data(self, table_model):
        """
        Fetch all rows from the specified table.
        """
        try:
            rows = self.db.query(table_model).all()
            print(f"Fetched {len(rows)} rows from database.")
            return rows
        except SQLAlchemyError as e:
            print(f"Error fetching data: {e}")
            return []

    def generate_embedding(self, text: str):
        """
        Generate embedding for a given text after tokenization and truncation.
        """
        # Tokenize and truncate the input text
        tokenized = TOKENIZER(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        truncated_text = TOKENIZER.decode(tokenized["input_ids"][0], skip_special_tokens=True)

        # Generate embeddings
        embedding_vector = self.embedding_model.encode(truncated_text)
        return embedding_vector.astype(np.float32)


    def process_and_save_embeddings(self, source_table, target_table, batch_size=4):
        """
        Process embeddings on CPU in small batches and save them to the target table.
        """
        rows = self.fetch_data(source_table)
        if not rows:
            print("No data to process.")
            return

        total_rows = len(rows)
        num_batches = (total_rows + batch_size - 1) // batch_size
        print(f"Processing {total_rows} rows in {num_batches} batches...")

        # Force CPU explicitly
        device = torch.device("cpu")
        

        for batch_idx in range(num_batches):  # Sequential processing, no tqdm
            print(f"\nProcessing batch {batch_idx + 1}/{num_batches}")
            batch_rows = rows[batch_idx * batch_size: (batch_idx + 1) * batch_size]

            texts = []
            for row in batch_rows:
                row_data = " ".join([
                    str(getattr(row, col) or "") for col in row.__table__.columns.keys()
                ]).strip()
                if row_data:
                    texts.append(row_data)

            print(f"Texts to process in batch {batch_idx + 1}: {len(texts)}")
            if not texts:
                print(f"Batch {batch_idx + 1} has no valid text data, skipping...")
                continue

            try:
                # Use CPU for embedding generation
                embeddings = self.embedding_model.encode(
                    texts, batch_size=batch_size, device=device
                )
                print(f"Embeddings generated successfully for batch {batch_idx + 1}")
            except Exception as e:
                print(f"Error generating embeddings for batch {batch_idx + 1}: {e}")
                break

            # Save embeddings to the database
            for i, row in enumerate(batch_rows):
                try:
                    new_embedding_entry = target_table(
                        code=row.code,
                        embedding=embeddings[i].astype(np.float32).tobytes()
                    )
                    self.db.add(new_embedding_entry)
                except Exception as e:
                    print(f"Error saving embedding for row {row.id}: {e}")
                    continue

            # Commit after each batch
            try:
                self.db.commit()
                print(f"Batch {batch_idx + 1}/{num_batches} committed successfully.")
            except Exception as e:
                print(f"Error committing batch {batch_idx + 1}: {e}")
                self.db.rollback()
                break


def main_embedding_process(db: Session, source_table, target_table):
    """
    Entry point to process embeddings.
    :param db: SQLAlchemy Session object
    :param source_table: SQLAlchemy table model for the source data
    :param target_table: SQLAlchemy table model to save embeddings
    """
    print("Initializing embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

    # Initialize the embedding processor
    processor = EmbeddingProcessor(db, embedding_model)

    # Process and save embeddings
    print("Starting embedding processing...")
    # row_text = "This is a test sentence for embedding generation."
    # embedding_vector = embedding_model.encode(row_text, device=device)
    # print(f"Test embedding shape: {embedding_vector.shape}")

    processor.process_and_save_embeddings(source_table, target_table)
    print("Embedding generation and saving complete.")
