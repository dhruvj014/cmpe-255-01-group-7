import asyncio
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import ReviewRequest, ReviewResponse
from inference import pipeline

app = FastAPI(
    title="Fake Review Detection API",
    description="CMPE-255 Group 7 — Multi-layer fake review detection pipeline",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health():
    return {"status": "ok", "message": "Fake Review Detection API is running"}


@app.post("/api/analyze", response_model=ReviewResponse)
async def analyze_review(request: ReviewRequest):
    try:
        data = request.model_dump()
        # Simulate multi-layer inference time (1.8 – 2.8 s)
        await asyncio.sleep(1.8 + random.random())
        result = pipeline.predict(data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
