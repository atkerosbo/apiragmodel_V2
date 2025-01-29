from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
import numpy as np
from database import get_db
from models import Products, Information, SeparateEmbeddingTables
from faiss_index import get_faiss_resources
from description_faiss_index import get_description_faiss_resources
from separate_faiss_index import get_separate_faiss_index_resources
from utils.tokens import verify_token, verify_key

from utils.chat_prompt_openai import summerize_answer


# Initialize FastAPI Router
router = APIRouter()

# Load SentenceTransformer model
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")

@router.post("/semantic-search", dependencies=[Depends(verify_token), Depends(verify_key)], status_code=200)
async def semantic_search(user_query: dict, db: Session = Depends(get_db)):
    """
    Perform semantic search to find similar products.
    """
    query = user_query
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # Step 1: Generate query embedding
        print("Generating query embedding...")
        query_embedding = embedding_model.encode([query], device="cpu").astype(np.float32)
        print(f"Query embedding shape: {query_embedding.shape}")  

        # Step 2: Retrieve FAISS resources
        print("Loading FAISS index...")
        index, product_codes = get_faiss_resources()
        print(f"FAISS index has {len(product_codes)} product codes.")

        # Step 3: Search FAISS index
        print("Searching FAISS index...")
        distances, indices = index.search(query_embedding, 5)
        print(f"Distances: {distances}")
        print(f"Indices: {indices}")

       
        # Step 4: Match product codes
        matched_codes = [
        product_codes[indices[0][i]]
        for i in range(len(distances[0]))
        if distances[0][i] < 0.9 and indices[0][i] < len(product_codes)
                ]
        if matched_codes == []:
            return {"Molim Vas detaljniji opis proizvoda."}

        # Step 5: Fetch product details
        print("Fetching products from the database...")
        results = db.query(Products).filter(Products.code.in_(matched_codes)).all()
        response = [{
            "code": product.code,
            "name": product.name,
            "product URL": product.prod_url,
            "description": product.description,
            "price": product.price,
        }
            for product in results
        ]
        print("Returning search results...")
        return response

    except Exception as e:
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    


@router.post("/semantic-search-descritption", dependencies=[Depends(verify_token), Depends(verify_key)], status_code=200)
async def semantic_search_description(user_query: dict, db: Session = Depends(get_db)):
    """
    Perform semantic search to find similar products.
    """
    query = user_query
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # Step 1: Generate query embedding
        print("Generating query embedding...")
        query_embedding = embedding_model.encode([query], device="cpu").astype(np.float32)
        print(f"Query embedding shape: {query_embedding.shape}")  

        # Step 2: Retrieve FAISS resources
        print("Loading FAISS index...")
        description_index, information_codes = get_description_faiss_resources()
        print(f"FAISS index has {len(information_codes)} information codes.")

        # Step 3: Search FAISS index
        # print("Searching FAISS index...")
        # distances, indices = description_index.search(query_embedding, 5)
        # print(f"Distances: {distances}")
        # print(f"Indices: {indices}")
        

       
        # # Step 4: Match information codes
        # matched_codes = [
        #     information_codes[indices[0][i]]
        #     for i in range(len(indices[0]))
        #     if indices[0][i] < len(information_codes)
        # ]

        print(f"FAISS index has {len(information_codes)} information codes.")
        print(f"FAISS index contains {description_index.ntotal} vectors.")
        print("Searching FAISS index...")

        try:
            distances, indices = description_index.search(query_embedding, 1)
            print(f"indices: {indices}, distances: {distances}")

            if len(indices[0]) == 0:
                print("No matches found.")
                matched_codes = []
            else:
                matched_codes = [
                    information_codes[idx]
                    for idx in indices[0]
                    if idx < len(information_codes)
                ]

            print(f"Matched codes: {matched_codes}")

        except Exception as e:
            matched_codes = []
            print(f"An error occurred: {e}")

        # Return or process matched_codes
        # return {"results": matched_codes}


        # Step 5: Fetch product details
        print("Fetching information from the database...")
        results = db.query(Information).filter(Information.code.in_(matched_codes)).all()
        response = [{
            "code": information.code,
            "link": information.link,
            "naslov":information.naslov,
            "opis": information.opis,
            
        }
            for information in results
        ]
        print("Returning search results...")
        #Summarize the response
        chunk_to_summerize = response[0]["opis"]
        summerized_response = summerize_answer(query, chunk_to_summerize)
        return {summerized_response}

    except Exception as e:
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    

@router.post("/semantic-search-separate", dependencies=[Depends(verify_token), Depends(verify_key)], status_code=200)
async def semantic_search_separate(user_query: dict, db: Session = Depends(get_db)):
    """
    Perform semantic search to find similar products.
    """
    query = user_query
    #print(f"Query: {query}")
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # Step 1: Generate query embedding
        print("Generating query embedding...")
        query_embedding = embedding_model.encode([query], device="cpu").astype(np.float32)
        print(f"Query embedding shape: {query_embedding.shape}")  

        # Step 2: Retrieve FAISS resources
        print("Loading FAISS index...")
        index, product_codes = get_separate_faiss_index_resources()
        print(f"FAISS index has {len(product_codes)} product codes.")

        # Step 3: Search FAISS index
        print("Searching FAISS index...")
        distances, indices = index.search(query_embedding, 5)
        print(f"Distances: {distances}")
        print(f"Indices: {indices}")

        #total_codes = [semantic_results_for_keywords, semnatic_results_for_query]
        # Step 4: Match product codes
        matched_codes = [
        product_codes[indices[0][i]]
        for i in range(len(distances[0]))
        if distances[0][i] < 0.9 and indices[0][i] < len(product_codes)
                ]
        if matched_codes == []:
            return {"Molim Vas detaljniji opis proizvoda."}
        print(f"Matched codes: {matched_codes}")
        # Step 5: Fetch product details
        print("Fetching products from the database...")
        results = db.query(Products).filter(Products.code.in_(matched_codes)).all()
        response = [{
            "code": product.code,
            "name": product.name,
            "product URL": product.prod_url,
            "description": product.description,
            "price": product.price,
        }
            for product in results
        ]
        print("Returning search results...")
        return response

    except Exception as e:
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    