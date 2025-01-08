from database import Base
from sqlalchemy import Column, Integer, String, LargeBinary
from sqlalchemy.sql.sqltypes import TIMESTAMP

class Data(Base):
    __tablename__ = "data"
    id = Column(Integer, primary_key=True, nullable=False)
    page_number = Column(Integer, nullable = False)
    page_char_count = Column(Integer, nullable = False)
    page_word_count =Column(Integer, nullable = False)
    page_sentence_count = Column(Integer, nullable = False)
    page_token_count = Column(Integer, nullable = False)
    text = Column(String, nullable=False)
    embedding = Column(LargeBinary, nullable=False)

class Products(Base):
    __tablename__ = "products"
    name = Column(String, nullable=True)
    prod_url = Column(String, nullable=True)
    code = Column(String,primary_key=True, nullable=False)    
    categories = Column(String, nullable=True)
    price = Column(String, nullable=True)
    label = Column(String, nullable=True)
    brand = Column(String, nullable=True)
    unit = Column(String, nullable=True)
    options = Column(String, nullable=True)
    description = Column(String, nullable=True)
    oe_numbers = Column(String, nullable=True)
    usage = Column(String, nullable=True)
    # description_embedding = Column(LargeBinary, nullable=True)


class Information(Base):
    __tablename__ = "information"
    code = Column(Integer, primary_key=True, nullable=False , autoincrement=True)
    link = Column(String, nullable=True)
    naslov = Column(String, nullable=True)
    opis = Column(String, nullable= True)


class EmbeddingsTable(Base):
    __tablename__ = "embeddings_table"
    code = Column(String, primary_key=True, index=True)  
    embedding = Column(LargeBinary, nullable=False)     
    


class InformationEmbeddings(Base):
    __tablename__ = "information_embeddings"
    code = Column(String, primary_key=True, index=True)  
    embedding = Column(LargeBinary, nullable=False)     
    
