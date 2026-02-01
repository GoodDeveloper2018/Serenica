"""
Chat API Router
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import List, Optional
import uuid

from app.models import Conversation, Session, User
from app.config import settings

router = APIRouter()

# ============================================================
# SCHEMAS
# ============================================================

class MessageRequest(BaseModel):
    session_id: str
    user_message: str

class MessageResponse(BaseModel):
    response: str
    topic: str
    confidence: float
    is_crisis: bool
    severity: int

# ============================================================
# ENDPOINTS
# ============================================================

async def process_message(
    user_id: str,
    message: str,
    session_id: str,
    db: AsyncSession,
    services: dict
):
    """Process user message through ML pipeline"""
    
    # Get session
    session = await db.get(Session, uuid.UUID(session_id))
    if not session or session.user_id != uuid.UUID(user_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get conversation history for context
    from sqlalchemy import select
    
    stmt = select(Conversation).where(
        Conversation.session_id == uuid.UUID(session_id)
    ).order_by(Conversation.message_index)
    
    result = await db.execute(stmt)
    previous_conversations = result.scalars().all()
    
    # Build conversation history
    conversation_history = [
        {"role": "user" if msg.user_message else "bot", "content": msg.user_message or msg.bot_response}
        for msg in previous_conversations[-10:]  # Last 10 messages
    ]
    
    # Process through ML pipeline
    ml_pipeline = services['ml_pipeline']
    rag_engine = services['rag_engine']
    safety_guard = services['safety_guard']
    
    # Get RAG context
    rag_context = await rag_engine.retrieve_context(message)
    
    # Process message
    result = await ml_pipeline.process_message(
        user_message=message,
        conversation_history=conversation_history,
        rag_context=rag_context
    )
    
    # Store in database
    new_message = Conversation(
        session_id=uuid.UUID(session_id),
        user_id=uuid.UUID(user_id),
        user_message=message,
        bot_response=result['response'],
        message_index=len(previous_conversations),
        detected_topic=result['topic'],
        metadata={
            "confidence": result['confidence'],
            "is_crisis": result['is_crisis'],
            "severity": result['severity']
        }
    )
    db.add(new_message)
    
    # Store safety flags if crisis detected
    if result['is_crisis']:
        from app.models import SafetyFlag
        safety_flag = SafetyFlag(
            conversation_id=new_message.id,
            user_id=uuid.UUID(user_id),
            flag_type=result['crisis_type'],
            severity=result['severity'],
            flagged_text=message
        )
        db.add(safety_flag)
    
    await db.commit()
    
    return MessageResponse(
        response=result['response'],
        topic=result['topic'],
        confidence=result['confidence'],
        is_crisis=result['is_crisis'],
        severity=result['severity']
    )

@router.post("/message", response_model=MessageResponse)
async def send_message(
    request: MessageRequest,
    user_id: str = None,
    db: AsyncSession = Depends(get_db)
):
    """Send message and get response"""
    # This will be called from HTTP endpoint
    pass

@router.get("/history/{session_id}")
async def get_conversation_history(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get conversation history for session"""
    from sqlalchemy import select
    
    stmt = select(Conversation).where(
        Conversation.session_id == uuid.UUID(session_id)
    ).order_by(Conversation.message_index)
    
    result = await db.execute(stmt)
    conversations = result.scalars().all()
    
    return [
        {
            "message_index": c.message_index,
            "user_message": c.user_message,
            "bot_response": c.bot_response,
            "topic": c.detected_topic,
            "timestamp": c.created_at
        }
        for c in conversations
    ]

# Helper for dependency injection
async def get_db() -> AsyncSession:
    """Get database session - should be imported from main.py"""
    pass
