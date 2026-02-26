"""
FastAPI routing for garment and outfit retrieval endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter(prefix="/api/v1/recommend", tags=["Recommendation"])

class RecommendationRequest(BaseModel):
    category: str
    preferred_style: Optional[str] = "casual"
    top_k: int = 5

class GarmentItem(BaseModel):
    item_id: str
    category: str
    match_score: float
    color_family: str

@router.post("/item", response_model=List[GarmentItem])
async def recommend_item(req: RecommendationRequest):
    """Retrieve top-K matching garment items."""
    return [
        GarmentItem(item_id=f"item_{i}", category=req.category, match_score=0.95 - (i * 0.05), color_family="Navy")
        for i in range(req.top_k)
    ]
