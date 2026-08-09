"""Entrypoint backend: python -m online.backend.api_server (chạy từ repo root).

Cấu trúc:
  schemas.py            - Pydantic request/response models
  resources.py          - AppResources + lifespan (init model/DB, degrade khi thiếu)
  utils.py              - helper thuần (timestamp, enrich ASR, Qdrant lookup)
  fusion.py             - RRF / SRRF fusion
  search_kis.py         - pipeline KIS
  search_temporal.py    - temporal enhanced + combinate
  search_intelligent.py - intelligent search
  routes.py             - HTTP endpoints (APIRouter)
"""
import os
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Phải đặt trước khi torch được import (qua chuỗi import bên dưới)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:64'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from online.backend.resources import lifespan
from online.backend.routes import router

logging.basicConfig(level=logging.INFO)

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3100,http://127.0.0.1:3100",
    ).split(",")
    if origin.strip()
]

app = FastAPI(
    title="VBS API",
    description="Visual Browsing System Backend API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
