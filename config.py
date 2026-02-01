"""
Configuration for Serenica
Environment variables and settings
"""

from typing import List
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://user:password@localhost:5432/serenica"
    )
    
    # JWT
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000"
    ]
    
    # ML Model
    MODEL_NAME: str = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")
    USE_OLLAMA: bool = os.getenv("USE_OLLAMA", "False").lower() == "true"
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")
    
    # Embeddings
    EMBEDDINGS_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    
    # Safety
    SAFETY_ENABLED: bool = True
    CRISIS_DETECTION_ENABLED: bool = True
    
    # RAG
    RAG_TOP_K: int = 3
    RAG_ENABLED: bool = True
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
