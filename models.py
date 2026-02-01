"""
Database models for Serenica
SQLAlchemy ORM models
"""

import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    Column, String, Text, Integer, Boolean, DateTime,
    ForeignKey, ARRAY, JSONB, Float
)
from sqlalchemy.dialects.postgresql import UUID, VECTOR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

# ============================================================
# USER & AUTHENTICATION
# ============================================================

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    username = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    is_therapist = Column(Boolean, default=False)
    profile_data = Column(JSONB, default={})
    
    # Relationships
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    preferences = relationship("UserPreference", back_populates="user", uselist=False, cascade="all, delete-orphan")
    goals = relationship("TherapyGoal", back_populates="user", cascade="all, delete-orphan")
    journal_entries = relationship("JournalEntry", back_populates="user", cascade="all, delete-orphan")

# ============================================================
# SESSIONS & CONVERSATIONS
# ============================================================

class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    session_metadata = Column(JSONB, default={})
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    conversations = relationship("Conversation", back_populates="session", cascade="all, delete-orphan")

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    user_message = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    message_index = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    detected_topic = Column(String(100))
    embedding = Column(VECTOR(384))  # Embedding vector
    metadata = Column(JSONB, default={})
    
    # Relationships
    session = relationship("Session", back_populates="conversations")
    user = relationship("User", back_populates="conversations")
    feedback = relationship("MessageFeedback", back_populates="conversation", cascade="all, delete-orphan")
    safety_flags = relationship("SafetyFlag", back_populates="conversation", cascade="all, delete-orphan")

# ============================================================
# FEEDBACK & RATINGS
# ============================================================

class MessageFeedback(Base):
    __tablename__ = "message_feedback"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    rating = Column(Integer)  # 1-5
    is_helpful = Column(Boolean)
    feedback_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="feedback")

# ============================================================
# SAFETY
# ============================================================

class SafetyFlag(Base):
    __tablename__ = "safety_flags"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    flag_type = Column(String(50))  # 'crisis', 'self_harm', etc.
    severity = Column(Integer)  # 1-5
    flagged_text = Column(Text)
    flag_timestamp = Column(DateTime, default=datetime.utcnow)
    is_resolved = Column(Boolean, default=False)
    resolution_notes = Column(Text)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="safety_flags")

# ============================================================
# USER PREFERENCES & GOALS
# ============================================================

class UserPreference(Base):
    __tablename__ = "user_preferences"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True)
    theme = Column(String(20), default="light")
    language = Column(String(10), default="en")
    notification_enabled = Column(Boolean, default=True)
    anonymous_mode = Column(Boolean, default=False)
    preferred_topics = Column(ARRAY(String))
    communication_style = Column(String(50), default="balanced")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="preferences")

class TherapyGoal(Base):
    __tablename__ = "therapy_goals"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    goal_title = Column(String(255), nullable=False)
    goal_description = Column(Text)
    goal_category = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    target_date = Column(DateTime)
    status = Column(String(20), default="active")  # active, completed, paused
    progress_percentage = Column(Integer, default=0)
    metadata = Column(JSONB, default={})
    
    # Relationships
    user = relationship("User", back_populates="goals")

# ============================================================
# JOURNAL
# ============================================================

class JournalEntry(Base):
    __tablename__ = "journal_entries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    entry_text = Column(Text, nullable=False)
    mood_score = Column(Integer)  # 1-10
    mood_tags = Column(ARRAY(String))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_private = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="journal_entries")

# ============================================================
# CRISIS RESOURCES
# ============================================================

class CrisisResource(Base):
    __tablename__ = "crisis_resources"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    resource_name = Column(String(255), nullable=False)
    resource_type = Column(String(50))  # hotline, website, app, center
    description = Column(Text)
    phone_number = Column(String(20))
    website_url = Column(String(500))
    country = Column(String(100))
    region = Column(String(100))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# ============================================================
# KNOWLEDGE BASE
# ============================================================

class TherapyKnowledgeBase(Base):
    __tablename__ = "therapy_knowledge_base"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    topic = Column(String(100))
    source = Column(String(255))
    therapist_name = Column(String(255))
    therapist_verified = Column(Boolean, default=False)
    embedding = Column(VECTOR(384))
    upvotes = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# ============================================================
# SYSTEM CONFIGURATION
# ============================================================

class SystemConfiguration(Base):
    __tablename__ = "system_configurations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    config_key = Column(String(255), unique=True, nullable=False)
    config_value = Column(JSONB, nullable=False)
    description = Column(Text)
    version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# ============================================================
# ANALYTICS
# ============================================================

class Analytics(Base):
    __tablename__ = "analytics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    event_type = Column(String(100), nullable=False)
    event_data = Column(JSONB, default={})
    timestamp = Column(DateTime, default=datetime.utcnow)

# ============================================================
# MODEL VERSIONING
# ============================================================

class ModelVersion(Base):
    __tablename__ = "model_versions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100))
    model_version = Column(String(50))
    model_path = Column(String(500))
    parameters = Column(JSONB)
    accuracy_metrics = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=False)

class InferenceLog(Base):
    __tablename__ = "inference_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_version_id = Column(UUID(as_uuid=True), ForeignKey("model_versions.id"))
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"))
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    inference_time_ms = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
