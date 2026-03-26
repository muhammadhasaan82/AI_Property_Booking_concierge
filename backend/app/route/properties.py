# route/properties.py
from typing import List, Optional
from fastapi import APIRouter
from pydantic import BaseModel
from app.components.search import property_search

router = APIRouter()

class PropertySearchIn(BaseModel):
    query_text: str = ""
    budget: Optional[float] = None
    amenities: Optional[List[str]] = None
    location: Optional[str] = None
    beds: Optional[int] = None

@router.post("/properties/search")
def search_properties(body: PropertySearchIn):
    results = property_search(
        query_text=body.query_text,
        budget=body.budget,
        amenities=body.amenities,
        location=body.location,
        beds=body.beds,
    )
    return {"results": results}

