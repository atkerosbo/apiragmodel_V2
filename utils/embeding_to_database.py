from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from models import Data, Products, Information
from database import get_db
import numpy as np
import os
import csv
import logging, traceback
from typing import Any, List, Optional
from sentence_transformers import SentenceTransformer

import json

logger = logging.getLogger(__name__)

embedder = SentenceTransformer("all-mpnet-base-v2")

def save_to_database(pages_data_with_embeddings: list, db: Session):
    try:
        for page in pages_data_with_embeddings:
            # Convert embedding to bytes if it's not already
            if isinstance(page['embedding'], list):
                embedding_bytes = np.array(page['embedding'], dtype=np.float32).tobytes()
            else:
                embedding_bytes = page['embedding']

            new_data = Data(
                page_number=page['page_number'],
                page_char_count=page['page_char_count'],
                page_word_count=page['page_word_count'],
                page_sentence_count=page['page_sentence_count'],
                page_token_count=page['page_token_count'],
                text=page['text'],
                embedding=embedding_bytes
            )
            db.add(new_data)
        
        db.commit()
        print("All data added successfully")
        return {"message": "All data saved successfully"}
    except Exception as e:
        db.rollback()
        print(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while saving data: {str(e)}")


def sanitize_code(code_value):
    """
    Clean and standardize the code value
    """
    if not code_value:
        return None
    
    # Normalize the code: strip whitespace, replace problematic characters
    cleaned_code = str(code_value).strip()
    
    # Replace unicode dash-like characters with standard dash
    cleaned_code = cleaned_code.replace('‚Äì', '-')
    
    # Remove extra whitespace
    cleaned_code = ' '.join(cleaned_code.split())
    
    return cleaned_code


def import_csv_to_database(csv_path: str, db: Session = Depends(get_db)):
    try:
        # Count imports
        import_count = 0
        skipped_count = 0
        
        # Read CSV and import data
        with open(csv_path, 'r', encoding='utf-8-sig') as csvfile:
            csvreader = csv.DictReader(csvfile, delimiter=',')
            
            # Print out fieldnames to double-check
            print("CSV Columns:", csvreader.fieldnames)
            
            for row in csvreader:
                try:
                    # Sanitize code
                    code = sanitize_code(row['code'])
                    
                    # Skip if code is None or empty
                    if not code:
                        skipped_count += 1
                        continue
                    
                    # Map CSV columns to Products model 
                    new_product = Products(
                        name=row['name'],
                        categories=row['categories'],
                        prod_url=row['prod_url'],
                        code=code,
                        price=row['price'],
                        label=row['label'],
                        brand=row['brand'],
                        unit=row['unit'],
                        options=row['options'],
                        description=row['description']
                    )
                    
                    db.add(new_product)
                    import_count += 1
                
                except Exception as row_error:
                    print(f"Error processing row: {row_error}")
                    print("Problematic row:", row)
                    skipped_count += 1
                    continue
            
            # Commit the transaction
            db.commit()
            print(f"Successfully imported {import_count} products")
            print(f"Skipped {skipped_count} products due to errors")
    
    except Exception as e:
        print(f"Overall import error: {e}")
        db.rollback()
    finally:
        db.close()


# Optional: If you want a function to directly import from JSON data

def safe_serialize(value: Any) -> str:
    """
    Safely serialize value to a string, handling various data types
    
    Args:
        value: Input value to be serialized
    
    Returns:
        str: Serialized value
    """
    try:
        # Handle lists and complex objects by converting to JSON string
        if isinstance(value, (list, dict)):
            return json.dumps(value)
        
        # Convert to string, handling None
        return str(value) if value is not None else ''
    except Exception as e:
        logger.warning(f"Serialization error for {value}: {e}")
        return ''


def save_products_to_database(json_path: str, db: Session):
    """
    Save products from JSON file to the database, skipping already uploaded products.
    
    Args:
        json_path (str): Path to the JSON file.
        db (Session): SQLAlchemy database session.
    """
    try:
        # Step 1: Load the JSON data
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        if not data or not isinstance(data, list):
            raise ValueError("JSON file must be a list of lists.")

        # Step 2: Extract keys from the first row
        keys = data[0]  # First list are the keys

        # Step 3: Iterate over rows, convert them to dictionaries, and save to the database
        for row in data[1:]:  # Skip the first row with keys
            if len(row) != len(keys):
                continue  # Skip invalid rows where keys and values do not match
            
            # Convert row to dictionary
            product_data = dict(zip(keys, row))

            # Check if product with the same code already exists
            existing_product = db.query(Products).filter_by(code=product_data.get('code', '')).first()
            if existing_product:
                print(f"Skipping product with code {product_data.get('code', '')}: Already exists.")
                continue

            # Create a new product instance
            new_product = Products(
                name=product_data.get('name', ''),
                prod_url=product_data.get('prod_url', ''),
                code=product_data.get('code', ''),
                categories=product_data.get('categories', ''),
                price=product_data.get('price', ''),
                label=product_data.get('label', ''),
                brand=product_data.get('brand', ''),
                unit=product_data.get('unit', ''),
                options=product_data.get('options', ''),
                description=product_data.get('description', ''),
                oe_numbers=product_data.get('OE_Numbers', ''),
                usage=product_data.get('usage', '')
            )

            # Add to session
            db.add(new_product)
        
        # Step 4: Commit the transaction
        db.commit()
        print("Products saved successfully.")
        # delete the JSON file
        os.remove(json_path)

    except Exception as e:
        db.rollback()
        print(f"Error: {str(e)}")
        raise



def save_desription_to_database(json_path: str, db: Session):
    try:
        # Step 1: Load the JSON data
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        if not data or not isinstance(data, list):
            raise ValueError("JSON file must be a list of lists.")

        # Step 2: Extract keys from the first row
        keys = data[0].split(',')

        # Step 3: Iterate over rows, convert them to dictionaries, and save to the database
        for row in data[1:]:  # Skip the first row with keys
            if len(row) != len(keys):
                continue  # Skip invalid rows where keys and values do not match
            
            # Convert row to dictionary
            site_description = dict(zip(keys, row))

            new_description = Information(
            
            link = site_description.get('link', ''),
            naslov = site_description.get('naslov', ''),
            opis = site_description.get('opis', '')
                )

            # Add to session
            db.add(new_description)

            # Add to session
                    
            # Step 4: Commit the transaction
            db.commit()
            print("Products saved successfully.")
            # delete the JSON file
            #os.remove(json_path)

    except Exception as e:
        db.rollback()
        print(f"Error: {str(e)}")
        raise