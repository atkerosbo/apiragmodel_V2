import numpy as np
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sqlalchemy.exc import SQLAlchemyError

import torch

# Device setup
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

    def fetch_data(self, table_model, filter_conditions=None):
        """
        Fetch rows from the specified table with optional filter conditions.
        """
        try:
            query = self.db.query(table_model)
            if filter_conditions:
                query = query.filter(*filter_conditions)
            rows = query.all()
            print(f"Fetched {len(rows)} rows from database.")
            return rows
        except SQLAlchemyError as e:
            print(f"Error fetching data: {e}")
            return []

    def generate_embedding(self, text: str):
        """
        Generate embedding for a given text after tokenization and truncation.
        """
        tokenized = TOKENIZER(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        truncated_text = TOKENIZER.decode(tokenized["input_ids"][0], skip_special_tokens=True)
        embedding_vector = self.embedding_model.encode(truncated_text)
        return embedding_vector.astype(np.float32)

    def save_embeddings(self, rows, columns, target_table):
        """
        Generate and save embeddings for specified columns in the target table.
        """
        for row in rows:
            for col in columns:
                text_data = str(getattr(row, col, "") or "").strip()
                if not text_data:
                    print(f"Row {row.code}, column {col} has no valid text data, skipping...")
                    continue

                try:
                    embedding_vector = self.generate_embedding(text_data)
                    new_embedding_entry = target_table(
                        code=row.code,  # Associate embedding with the product
                        embedding=embedding_vector.tobytes()
                    )
                    self.db.add(new_embedding_entry)
                    print(f"Embedding generated and added for row {row.code}, column {col}.")
                except Exception as e:
                    print(f"Error generating/saving embedding for row {row.code}, column {col}: {e}")
                    continue

        try:
            self.db.commit()
            print("All embeddings committed successfully.")
        except Exception as e:
            print(f"Error committing embeddings: {e}")
            self.db.rollback()

    def process_columns_and_save(self, source_table, target_table, columns, filter_conditions=None):
        """
        Process embeddings for specific columns of rows from the source table
        and save them to the target table.
        """
        rows = self.fetch_data(source_table, filter_conditions)
        if not rows:
            print("No data to process.")
            return

        print(f"Processing embeddings for columns: {columns}")
        self.save_embeddings(rows, columns, target_table)

def main_embedding_process(db: Session, source_table, target_table):
    """
    Entry point to process embeddings.
    """
    print("Initializing embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    processor = EmbeddingProcessor(db, embedding_model)

    print("Starting embedding processing...")
    processor.process_columns_and_save(
        source_table=source_table,
        target_table=target_table,
        columns=["name", "description", "options", "usage"]
    )
    print("Embedding generation and saving complete.")

def separate_emb_process(db: Session, source_table, target_table):
    """
    Entry point for processing specific columns.
    """
    print("Initializing embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    processor = EmbeddingProcessor(db, embedding_model)

    columns_to_embed = ["name", "description", "options"]
    processor.process_columns_and_save(
        source_table=source_table,
        target_table=target_table,
        columns=columns_to_embed
    )
