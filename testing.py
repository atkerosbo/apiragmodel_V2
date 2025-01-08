import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# 1. Set up SQLite database
conn = sqlite3.connect('embeddings.db')
cursor = conn.cursor()

# Create a table to store text and embeddings
cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY,
        text TEXT,
        embedding BLOB
    )
''')

# 2. Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose an appropriate model

# Sample data
texts = [
    "This is a sample sentence.",
    "Another example text.",
    "Python is a great programming language.",
]

# Generate embeddings
embeddings = model.encode(texts)

# 3. Store embeddings in the database
for text, embedding in zip(texts, embeddings):
    cursor.execute(
        "INSERT INTO documents (text, embedding) VALUES (?, ?)",
        (text, embedding.tobytes())
    )

conn.commit()

# 4. Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the index
index.add(embeddings)

# Save the index
faiss.write_index(index, "faiss_index.bin")

# Example: Perform a similarity search
query = "Python programming"
query_embedding = model.encode([query])[0]

k = 2  # Number of nearest neighbors to retrieve
distances, indices = index.search(np.array([query_embedding]), k)

print(f"Top {k} similar documents for query '{query}':")
for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"{i+1}. Distance: {distance:.4f}, Text: {texts[idx]}")

# Close the database connection
conn.close()