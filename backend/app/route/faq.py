# route/FAQ.py
from fastapi import APIRouter, Query
from app.services.faq import faq_lookup

router = APIRouter()

@router.get("/faq")
def get_faq(question: str = Query(..., min_length=2)):
    ans = faq_lookup(question)
    return {"answer": ans, "source": "db" if ans else "llm_fallback"}

