"""
Serenica FastAPI Backend
Modern therapy chatbot with RAG, safety detection, and personalization
"""

import os
from contextlib import asynccontextmanager
from typing import Optional
import logging

from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
import uvicorn
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# Import routers and services
from app.api import chat, users, sessions, feedback, journal, goals, knowledge_base, admin
from app.services.ml_pipeline import MLPipeline
from app.services.safety_guard import SafetyGuard
from app.services.rag_engine import RAGEngine
from app.database import Base
from app.config import settings
from app.websocket_manager import ConnectionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# LIFESPAN: Startup & Shutdown
# ============================================================

async def init_db():
    """Initialize database and create tables"""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✓ Database initialized")
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {e}")
        raise

async def load_models():
    """Load ML models on startup"""
    try:
        logger.info("Loading ML pipeline...")
        ml_pipeline = MLPipeline()
        await ml_pipeline.initialize()
        logger.info("✓ ML pipeline loaded")
        
        logger.info("Loading safety guardrails...")
        safety_guard = SafetyGuard()
        await safety_guard.initialize()
        logger.info("✓ Safety guardrails loaded")
        
        logger.info("Loading RAG engine...")
        rag_engine = RAGEngine()
        await rag_engine.initialize()
        logger.info("✓ RAG engine loaded")
        
        return {
            'ml_pipeline': ml_pipeline,
            'safety_guard': safety_guard,
            'rag_engine': rag_engine
        }
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle"""
    # Startup
    await init_db()
    app.state.services = await load_models()
    logger.info("Serenica backend started successfully")
    
    yield
    
    # Shutdown
    logger.info("Serenica backend shutting down...")
    await engine.dispose()

# ============================================================
# DATABASE SETUP
# ============================================================

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_pre_ping=True,
)

async_session = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

async def get_db() -> AsyncSession:
    """Get database session"""
    async with async_session() as session:
        yield session

# ============================================================
# APP INITIALIZATION
# ============================================================

app = FastAPI(
    title="Serenica API",
    description="AI-powered therapy chatbot backend",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# DEPENDENCY INJECTION
# ============================================================

security = HTTPBearer()

# ============================================================
# ROUTERS
# ============================================================

app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(feedback.router, prefix="/api/feedback", tags=["feedback"])
app.include_router(journal.router, prefix="/api/journal", tags=["journal"])
app.include_router(goals.router, prefix="/api/goals", tags=["goals"])
app.include_router(knowledge_base.router, prefix="/api/kb", tags=["knowledge_base"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])

# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "services": {
            "database": "connected",
            "ml_pipeline": "ready",
            "safety_guard": "active",
            "rag_engine": "ready"
        }
    }

# ============================================================
# WEBSOCKET FOR REAL-TIME CHAT
# ============================================================

manager = ConnectionManager()

@app.websocket("/ws/chat/{user_id}")
async def websocket_chat(websocket: WebSocket, user_id: str, db: AsyncSession = Depends(get_db)):
    """WebSocket endpoint for real-time chat"""
    await manager.connect(user_id, websocket)
    try:
        while True:
            data = await websocket.receive_json()
            
            # Process message through ML pipeline with RAG and safety checks
            response = await chat.process_message(
                user_id=user_id,
                message=data.get("message"),
                session_id=data.get("session_id"),
                db=db,
                services=app.state.services
            )
            
            await manager.send_personal_message(user_id, response)
    except WebSocketDisconnect:
        manager.disconnect(user_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.send_personal_message(user_id, {
            "error": "Processing error occurred",
            "type": "error"
        })

# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Serenica",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
