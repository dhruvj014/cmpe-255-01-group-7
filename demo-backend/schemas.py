from pydantic import BaseModel, Field
from typing import List


class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=10, description="The review text")
    rating: int = Field(..., ge=1, le=5, description="Star rating (1-5)")
    tenure_days: int = Field(..., ge=0, description="Age of the account in days")
    review_count: int = Field(..., ge=1, description="Total reviews by the user")
    seller_concentration: float = Field(..., ge=0.0, le=1.0, description="Fraction of reviews for a single seller")
    burst_score: int = Field(..., ge=0, description="Max reviews in a 7-day window")


class ReviewResponse(BaseModel):
    is_spam: bool
    spam_probability: float
    behavioral_risk_tier: str
    top_factors: List[str]
    layer_scores: dict
