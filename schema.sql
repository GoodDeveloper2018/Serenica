-- Serenica Modern Architecture Schema
-- PostgreSQL with pgvector for semantic search
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS uuid - ossp;
-- ============================================================
-- 1. USERS & AUTHENTICATION
-- ============================================================
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    is_therapist BOOLEAN DEFAULT FALSE,
    profile_data JSONB DEFAULT '{}' -- Stores additional profile info
);
-- ============================================================
-- 2. SESSIONS & CONVERSATIONS
-- ============================================================
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_metadata JSONB DEFAULT '{}',
    -- Mood, context, etc.
    is_active BOOLEAN DEFAULT TRUE
);
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    user_message TEXT NOT NULL,
    bot_response TEXT NOT NULL,
    message_index INTEGER NOT NULL,
    -- Order in conversation
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    detected_topic VARCHAR(100),
    -- anxiety, depression, relationship, etc.
    embedding vector(384),
    -- Sentence transformer embeddings (all-MiniLM-L6-v2)
    metadata JSONB DEFAULT '{}' -- token_count, confidence, etc.
);
CREATE INDEX idx_conversations_session ON conversations(session_id);
CREATE INDEX idx_conversations_user ON conversations(user_id);
CREATE INDEX idx_conversations_embedding ON conversations USING ivfflat (embedding vector_cosine_ops);
-- ============================================================
-- 3. FEEDBACK & RATINGS
-- ============================================================
CREATE TABLE message_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    rating INTEGER CHECK (
        rating >= 1
        AND rating <= 5
    ),
    -- 1-5 stars
    is_helpful BOOLEAN,
    -- Thumbs up/down
    feedback_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_feedback_user ON message_feedback(user_id);
-- ============================================================
-- 4. SAFETY & CRISIS DETECTION
-- ============================================================
CREATE TABLE safety_flags (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    flag_type VARCHAR(50),
    -- 'crisis', 'self_harm', 'suicidal', 'concerning'
    severity INTEGER CHECK (
        severity >= 1
        AND severity <= 5
    ),
    -- 1-5 severity
    flagged_text TEXT,
    flag_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_resolved BOOLEAN DEFAULT FALSE,
    resolution_notes TEXT
);
CREATE INDEX idx_safety_user ON safety_flags(user_id);
CREATE INDEX idx_safety_unresolved ON safety_flags(is_resolved)
WHERE is_resolved = FALSE;
-- ============================================================
-- 5. USER PREFERENCES & GOALS
-- ============================================================
CREATE TABLE user_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    theme VARCHAR(20) DEFAULT 'light',
    -- light, dark
    language VARCHAR(10) DEFAULT 'en',
    notification_enabled BOOLEAN DEFAULT TRUE,
    anonymous_mode BOOLEAN DEFAULT FALSE,
    preferred_topics TEXT [],
    -- Array of topic interests
    communication_style VARCHAR(50) DEFAULT 'balanced',
    -- formal, casual, balanced
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE therapy_goals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    goal_title VARCHAR(255) NOT NULL,
    goal_description TEXT,
    goal_category VARCHAR(100),
    -- anxiety_management, relationship_skills, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    target_date DATE,
    status VARCHAR(20) DEFAULT 'active',
    -- active, completed, paused
    progress_percentage INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}' -- Tracking additional goal data
);
CREATE INDEX idx_goals_user ON therapy_goals(user_id);
-- ============================================================
-- 6. JOURNAL & MOOD TRACKING
-- ============================================================
CREATE TABLE journal_entries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    entry_text TEXT NOT NULL,
    mood_score INTEGER CHECK (
        mood_score >= 1
        AND mood_score <= 10
    ),
    -- 1-10 scale
    mood_tags TEXT [],
    -- happy, anxious, sad, angry, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_private BOOLEAN DEFAULT TRUE
);
CREATE INDEX idx_journal_user ON journal_entries(user_id);
CREATE INDEX idx_journal_date ON journal_entries(user_id, created_at);
-- ============================================================
-- 7. CRISIS RESOURCES DATABASE
-- ============================================================
CREATE TABLE crisis_resources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resource_name VARCHAR(255) NOT NULL,
    resource_type VARCHAR(50),
    -- hotline, website, app, center
    description TEXT,
    phone_number VARCHAR(20),
    website_url VARCHAR(500),
    country VARCHAR(100),
    region VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- ============================================================
-- 8. KNOWLEDGE BASE / THERAPY CONTENT
-- ============================================================
CREATE TABLE therapy_knowledge_base (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    topic VARCHAR(100),
    -- anxiety, depression, relationship, etc.
    source VARCHAR(255),
    -- Where this came from (therapist, research, etc.)
    therapist_name VARCHAR(255),
    therapist_verified BOOLEAN DEFAULT FALSE,
    embedding vector(384),
    -- For semantic search
    upvotes INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_kb_topic ON therapy_knowledge_base(topic);
CREATE INDEX idx_kb_embedding ON therapy_knowledge_base USING ivfflat (embedding vector_cosine_ops);
-- ============================================================
-- 9. CONVERSATION TEMPLATES / SYSTEM PROMPTS
-- ============================================================
CREATE TABLE system_configurations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_key VARCHAR(255) UNIQUE NOT NULL,
    config_value JSONB NOT NULL,
    description TEXT,
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- Example configs: safety_guardrails, response_style, model_params, etc.
-- ============================================================
-- 10. ANALYTICS & METRICS
-- ============================================================
CREATE TABLE analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE
    SET NULL,
        event_type VARCHAR(100),
        -- conversation_started, message_sent, goal_created, etc.
        event_data JSONB DEFAULT '{}',
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_analytics_user ON analytics(user_id);
CREATE INDEX idx_analytics_type ON analytics(event_type);
-- ============================================================
-- 11. MODEL VERSIONS & INFERENCE LOGS
-- ============================================================
CREATE TABLE model_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100),
    -- mistral, llama, custom-fine-tuned
    model_version VARCHAR(50),
    model_path VARCHAR(500),
    parameters JSONB,
    -- Temperature, top_p, etc.
    accuracy_metrics JSONB,
    -- Performance on test set
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT FALSE
);
CREATE TABLE inference_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_version_id UUID REFERENCES model_versions(id),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    input_tokens INTEGER,
    output_tokens INTEGER,
    inference_time_ms INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- ============================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================
-- User activity summary
CREATE VIEW user_activity_summary AS
SELECT u.id,
    u.username,
    COUNT(DISTINCT s.id) as total_sessions,
    COUNT(DISTINCT c.id) as total_messages,
    MAX(c.created_at) as last_active,
    COUNT(DISTINCT jj.id) as journal_entries,
    COUNT(DISTINCT tg.id) as active_goals
FROM users u
    LEFT JOIN sessions s ON u.id = s.user_id
    LEFT JOIN conversations c ON u.id = c.user_id
    LEFT JOIN journal_entries jj ON u.id = jj.user_id
    LEFT JOIN therapy_goals tg ON u.id = tg.user_id
    AND tg.status = 'active'
GROUP BY u.id,
    u.username;
-- Safety incidents requiring attention
CREATE VIEW safety_incidents_pending AS
SELECT sf.id,
    sf.user_id,
    u.username,
    sf.flag_type,
    sf.severity,
    sf.flagged_text,
    sf.flag_timestamp,
    c.user_message,
    c.created_at as message_time
FROM safety_flags sf
    JOIN users u ON sf.user_id = u.id
    JOIN conversations c ON sf.conversation_id = c.id
WHERE sf.is_resolved = FALSE
ORDER BY sf.severity DESC,
    sf.flag_timestamp DESC;
-- Topic distribution
CREATE VIEW topic_distribution AS
SELECT detected_topic,
    COUNT(*) as message_count,
    AVG(
        EXTRACT(
            EPOCH
            FROM (updated_at - created_at)
        )
    ) as avg_response_time_sec
FROM conversations
GROUP BY detected_topic
ORDER BY message_count DESC;