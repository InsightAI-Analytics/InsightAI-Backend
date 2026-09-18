"""
InsightAI FastAPI application entry point.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routers import files, query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(
    title="InsightAI",
    description="AI-Powered Data Q&A API — upload CSV/Excel, ask questions, get answers.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(files.router)
app.include_router(query.router)


@app.get("/", tags=["health"])
async def root():
    return {"service": "InsightAI", "status": "running", "version": "1.0.0"}


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}
