from pydantic import BaseModel

class UserQuery(BaseModel):
    query: str


class SearchQuery(BaseModel):
    query: str
    top_k: int = 5