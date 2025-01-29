from sentence_transformers import SentenceTransformer
from fastapi import Depends, APIRouter, HTTPException
from utils.tokens import verify_key, verify_token
from sqlalchemy.orm import Session
from database import get_db
from dotenv import load_dotenv
from utils.chat_prompt_openai import get_keywords_with_openai, get_type_of_query, summerize_answer, chat_with_context
from routes.semantic_search import semantic_search, semantic_search_description, semantic_search_separate
import faiss
from models import EmbeddingsTable, Products, Information
from typing import List
load_dotenv()

router = APIRouter()

# Load the embedding model and FAISS index globally
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")

# FAISS index setup
INDEX_FILE = "index.faiss"
product_codes = []  # Global variable to hold product codes in the same order as FAISS index
index = None

global_query_thread: List[str] =[]


########### LOAD INDEX FOR SEMANTIC SEARCH ############

def load_faiss_index(db: Session):
    global index, product_codes
    print("Loading FAISS index...")
    index = faiss.read_index(INDEX_FILE)
    
    # Load product codes
    product_codes = [
        row.code for row in db.query(EmbeddingsTable).all()
    ]
    print(f"Loaded FAISS index with {len(product_codes)} product codes.")

########### FORMATING THE RESULTS ############

def format_product_results(products):
    return [
        {
            "code": product.code,
            "name": product.name,
            "product URL": product.prod_url,
            "description": product.description,
            "price": product.price,
        }
        for product in products
    ]

 ########### ENDPOINT OF THE API WITHOUT PRELOADED DATA############

# @router.post("/ragchat", dependencies=[Depends(verify_token), Depends(verify_key)], status_code=200)
# async def rag_chat(user_query: dict, db: Session = Depends(get_db)):
#     query = user_query.get("query")
#     if not query:
#         raise HTTPException(status_code=400, detail="Query cannot be empty.")

#     query_type = get_type_of_query(query)
#     print(f"Query type detected: {query_type}")

#     # Extract keywords
#     keywords = get_keywords_with_openai(query)
#     if not isinstance(keywords, list):
#         keywords = [kw.strip() for kw in keywords.split(",")]

#     print(f"Extracted keywords: {keywords}")

#     try:
#         for keyword in keywords:
#             print(f"Checking keyword: {keyword}")

#             # Check in product codes
#             product = db.query(Products).filter(Products.code == keyword).first()
#             if product:
#                 print(f"Match found in product codes for keyword: {keyword}")
#                 return {"answer": format_product_results([product])}

#             # Check in labels
#             product = db.query(Products).filter(Products.label == keyword).first()
#             if product:
#                 print(f"Match found in labels for keyword: {keyword}")
#                 return {"answer": format_product_results([product])}

#             # Check in options (wildcard search)
#             products = db.query(Products).filter(Products.options.ilike(f"%{keyword}%")).all()
#             if products:
#                 print(f"Match found in product options for keyword: {keyword}")
#                 return {"answer": format_product_results(products)}

#             # Check in product names (wildcard search)
#             products = db.query(Products).filter(Products.name.ilike(f"%{keyword}%")).all()
#             if products:
#                 print(f"Match found in product names for keyword: {keyword}")
#                 return {"answer": format_product_results(products)}

#             # Check in descriptions (wildcard search)
#             products = db.query(Products).filter(Products.description.ilike(f"%{keyword}%")).all()
#             if products:
#                 print(f"Match found in product descriptions for keyword: {keyword}")
#                 return {"answer": format_product_results(products)}

#             # Check in general information
#             info = db.query(Information).filter(Information.opis.ilike(f"%{keyword}%")).first()
#             if info:
#                 print(f"Match found in general information for keyword: {keyword}")
#                 info_shortened = summerize_answer(query, info.opis)
#                 return {"answer": info_shortened}

#         # Semantic search fallback
#         print("No direct matches found, falling back to semantic search.")
#         if query_type == "Product":
#             keywords_for_semantic = {"query": " ".join(keywords), "top_k": 3}
#             return await {"answer": semantic_search(user_query=keywords_for_semantic, db=db)}

#         elif query_type == "General":
#             keywords_for_semantic = {"query": " ".join(keywords), "top_k": 1}
#             return await {"answer": semantic_search_description(user_query=keywords_for_semantic, db=db)}

#         raise HTTPException(status_code=400, detail="Unknown query type.")

#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")



########### ENDPOINT OF THE API VERSION WITH PRELOADED DATA ############

@router.post("/ragchat", dependencies=[Depends(verify_token), Depends(verify_key)], status_code=200)
async def rag_chat(user_query: dict, db: Session = Depends(get_db)):
    user_query = user_query.get("query")
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    global_query_thread.append({"role": "user", "content": user_query})

    query_type = get_type_of_query(user_query)
    print(f"Query type detected: {query_type}")

   
    
    # Extract keywords
    keywords = get_keywords_with_openai(user_query)
    if not isinstance(keywords, list):
        keywords = [kw.strip() for kw in keywords.split(",")]

    # Log keywords for debugging
    print(f"Extracted keywords: {keywords}")

    # Pre-fetch data
    product_codes = {row.code for row in db.query(Products).all()}
    labels = {row.label for row in db.query(Products).all()}
      
    
    try:
        for keyword in keywords:
            print(f"Checking keyword: {keyword}")

            if keyword in product_codes:
                print("Match found in product codes")
                products = db.query(Products).filter(Products.code == keyword).all()
                return {"answer": format_product_results(products)}
            #print("Checking in labels")
            if keyword in labels:
                print("Match found in labels")
                products = db.query(Products).filter(Products.label == keyword).all()
                return {"answer": format_product_results(products)}
         
        # Semantic search fallback
        if query_type == "Product":
            keywords_for_semantic = {"query": " ".join(keywords), "top_k": 3}
            print("Fallback to semantic search for product")
            returning_answer = await semantic_search_separate(user_query=keywords_for_semantic, db=db)
            # returning_answer = []
            # for keyword in keywords:
            #     answer_for_keyword = await semantic_search_separate(user_query=keyword, db=db)
            #     returning_answer.extend(answer_for_keyword)
            global_query_thread.append({"role": "system", "answer": returning_answer})
            response_with_context = chat_with_context(global_query_thread)
            print(response_with_context)
            return {"answer": returning_answer}

        elif query_type == "General":
            keywords_for_semantic = {"query": " ".join(keywords), "top_k": 1}
            print("Fallback to semantic search for general query")
            returning_answer = await semantic_search_description(user_query=keywords_for_semantic, db=db)
            global_query_thread.append({"role": "system", "answer": returning_answer})
            return {"answer": returning_answer}

        print("No matching query type found.")
        raise HTTPException(status_code=400, detail="Unknown query type.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    

@router.get("/all_queries", dependencies=[Depends(verify_token), Depends(verify_key)], status_code=200)
async def get_all_queries():
    """
    Endpoint to retrieve all stored queries for future reference.
    """
    return global_query_thread