# Serenica v2.0 - Setup & Getting Started Guide

## Overview

Serenica is a modern, full-stack AI-powered therapy chatbot with:

- **FastAPI** backend with async inference
- **PostgreSQL + pgvector** for semantic search
- **Mistral 7B/Llama** for response generation
- **LangChain RAG** for knowledge retrieval
- **Real-time WebSockets** for chat
- **Safety guardrails** for crisis detection
- **Modern data pipeline** for continuous improvement

---

## Prerequisites

### System Requirements

- **Python**: 3.11+
- **Database**: PostgreSQL 15+ (with pgvector)
- **RAM**: 16GB+ (for ML models)
- **Storage**: 50GB+ (for model weights)

### Recommended Setup

- **OS**: Ubuntu 22.04 LTS or macOS 13+
- **GPU**: NVIDIA CUDA 12.x (optional but recommended)
- **Docker**: For containerization

---

## Step 1: Database Setup

### 1.1 Install PostgreSQL

**Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql
```

**macOS:**

```bash
brew install postgresql
brew services start postgresql
```

**Windows:**
Download from: https://www.postgresql.org/download/windows/

### 1.2 Create Database

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE serenica;

# Create user
CREATE USER serenica_user WITH PASSWORD 'secure_password';

# Grant privileges
GRANT ALL PRIVILEGES ON DATABASE serenica TO serenica_user;

# Exit
\q
```

### 1.3 Enable pgvector Extension

```bash
# Connect to the serenica database
psql -U serenica_user -d serenica

# Enable pgvector
CREATE EXTENSION IF NOT EXISTS pgvector;
CREATE EXTENSION IF NOT EXISTS uuid-ossp;

# Verify
\dx
```

### 1.4 Run Schema

```bash
# Connect and run schema
psql -U serenica_user -d serenica -f schema.sql
```

---

## Step 2: Python Environment Setup

### 2.1 Create Virtual Environment

```bash
# Navigate to project directory
cd c:\Users\arshp\OneDrive\Desktop\Work\serenica-app\Serenica

# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2.2 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: If CUDA available, install GPU version:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Step 3: ML Model Setup

### Option A: Using Ollama (Recommended for Local Dev)

```bash
# Install Ollama
# Visit: https://ollama.ai

# Start Ollama service
ollama serve

# In another terminal, download Mistral
ollama pull mistral

# Test
curl http://localhost:11434/api/generate -d '{
  "model": "mistral",
  "prompt": "Hello"
}'
```

### Option B: Local Transformers (GPU Required)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

---

## Step 4: Configuration

### 4.1 Create .env file

```bash
# Copy template
cp .env.example .env

# Edit with your settings
nano .env
```

### 4.2 Key Configuration

```
DATABASE_URL=postgresql+asyncpg://serenica_user:secure_password@localhost:5432/serenica
USE_OLLAMA=True
OLLAMA_MODEL=mistral
SECRET_KEY=your-super-secret-key-here
```

---

## Step 5: Data Loading

### 5.1 Prepare Your Data

The `data_loader.py` provides a complete pipeline. Expected CSV format:

```csv
question,answer,topic,source,therapist_name,therapist_verified
"How do I manage anxiety?","Try box breathing and progressive relaxation...",anxiety,"Research","Dr. Jane Smith, LMFT",true
```

### 5.2 Load Data

```python
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from data_loader import DataImporter
from ml_pipeline import MLPipeline

async def load_data():
    # Initialize engine
    engine = create_async_engine(
        "postgresql+asyncpg://serenica_user:password@localhost/serenica"
    )
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Initialize ML pipeline
    ml_pipeline = MLPipeline()
    await ml_pipeline.initialize()

    # Import data
    importer = DataImporter(ml_pipeline)
    async with async_session() as db:
        stats = await importer.import_csv(
            file_path="counselchat-data.csv",
            db=db,
            skip_validation=False
        )
        print(f"Loaded {stats['summary']['total_imported']} records")

# Run
asyncio.run(load_data())
```

### 5.3 Verify Import

```sql
-- Check counts
SELECT COUNT(*) FROM therapy_knowledge_base;

-- Check topics distribution
SELECT topic, COUNT(*) FROM therapy_knowledge_base GROUP BY topic;

-- Check embeddings
SELECT COUNT(*) FROM therapy_knowledge_base WHERE embedding IS NOT NULL;
```

---

## Step 6: Start the Application

### 6.1 Initialize Models

```bash
# In first terminal
ollama serve
```

### 6.2 Start FastAPI Server

```bash
# In second terminal
python main.py
```

Expected output:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
✓ Database initialized
✓ ML pipeline loaded
✓ Safety guardrails loaded
✓ RAG engine loaded
```

### 6.3 Check Health

```bash
curl http://localhost:8000/health
```

Response:

```json
{
  "status": "healthy",
  "version": "2.0.0",
  "services": {
    "database": "connected",
    "ml_pipeline": "ready",
    "safety_guard": "active",
    "rag_engine": "ready"
  }
}
```

### 6.4 View API Documentation

Open browser to: **http://localhost:8000/docs**

---

## Step 7: Test Chat

### Using WebSocket (Real-time)

```python
import asyncio
import websockets
import json

async def test_chat():
    uri = "ws://localhost:8000/ws/chat/test-user-123"

    async with websockets.connect(uri) as websocket:
        # Send message
        await websocket.send(json.dumps({
            "session_id": "session-456",
            "message": "I'm feeling anxious about work"
        }))

        # Receive response
        response = await websocket.recv()
        print("Bot:", json.loads(response))

asyncio.run(test_chat())
```

### Using HTTP REST

```bash
curl -X POST http://localhost:8000/api/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session-456",
    "user_message": "How do I manage anxiety?"
  }'
```

---

## Step 8: Data Import Interface (Boilerplate)

### 8.1 Get Template

```bash
curl http://localhost:8000/api/admin/import-template
```

### 8.2 Example Usage

```python
from data_loader import DataImporter

# Get template
template = DataImporter.get_import_template()

# Save sample CSV
with open("sample_data.csv", "w") as f:
    f.write(template['sample'])

# Schema validation
print(template['schema'])
```

### 8.3 Admin Endpoint (Example)

```python
# POST /api/admin/import-csv
# Expects multipart form data with CSV file

@app.post("/api/admin/import-csv")
async def import_data(
    file: UploadFile = File(...),
    skip_validation: bool = False,
    current_user: User = Depends(get_current_user)
):
    if not current_user.is_therapist:
        raise HTTPException(status_code=403)

    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        content = await file.read()
        tmp.write(content)

        importer = DataImporter(app.state.services['ml_pipeline'])
        async with AsyncSession(engine) as db:
            stats = await importer.import_csv(tmp.name, db, skip_validation)

        return stats
```

---

## Data Requirements Summary

| Volume     | QA Pairs | Quality | Timeline   | Cost     |
| ---------- | -------- | ------- | ---------- | -------- |
| MVP        | 2,000    | 70%     | 2-3 weeks  | Free-$5K |
| Production | 5,000    | 85%     | 6-8 weeks  | $10-30K  |
| Optimal    | 10,000+  | 90%+    | 3-4 months | $50-100K |

**Recommended Approach:**

1. Start with existing CounselChat dataset (~2K)
2. Add 500-1000 custom Q&A pairs
3. Partner with 2-3 therapists for review
4. Launch with MVP (2-3 weeks)
5. Expand to production over next 6 weeks

See `DATA_REQUIREMENTS.py` for full strategy.

---

## Docker Deployment

### Quick Start with Docker Compose

```yaml
version: "3.9"

services:
  db:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_DB: serenica
      POSTGRES_USER: serenica_user
      POSTGRES_PASSWORD: secure_password
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql+asyncpg://serenica_user:secure_password@db:5432/serenica
      USE_OLLAMA: "True"
      OLLAMA_MODEL: "mistral"
    depends_on:
      - db
      - redis
      - ollama
    command: python main.py

volumes:
  ollama_data:
```

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop
docker-compose down
```

---

## Monitoring & Debugging

### View Logs

```bash
# Tail logs
tail -f app.log

# Filter by level
grep ERROR app.log
```

### Database Queries

```sql
-- Recent conversations
SELECT * FROM conversations ORDER BY created_at DESC LIMIT 10;

-- Crisis flags
SELECT * FROM safety_flags WHERE is_resolved = FALSE;

-- RAG usage
SELECT detected_topic, COUNT(*) FROM conversations GROUP BY detected_topic;

-- User activity
SELECT username, COUNT(*) as messages FROM users u
JOIN conversations c ON u.id = c.user_id
GROUP BY u.id, u.username;
```

### Performance Monitoring

```bash
# Check inference time
SELECT AVG(inference_time_ms) FROM inference_logs;

# Model accuracy
SELECT * FROM model_versions WHERE is_active = TRUE;
```

---

## Common Issues & Solutions

### "psycopg2 connection refused"

- Ensure PostgreSQL is running: `sudo systemctl status postgresql`
- Check DATABASE_URL in .env
- Verify credentials

### "Model download stuck"

- Check internet connection
- Increase timeout: `pip install transformers --upgrade`
- Use Ollama instead of local transformers

### "Out of memory (OOM)"

- Use 4-bit quantization
- Reduce batch size
- Use smaller model (DistilMistral)
- Upgrade RAM or use GPU

### "Embeddings not indexing"

- Check pgvector extension: `psql -d serenica -c "\dx"`
- Verify VECTOR column type in schema
- Manually re-index: `REINDEX INDEX idx_kb_embedding;`

---

## Next Steps

1. ✅ **Load initial data** (counselchat dataset)
2. ✅ **Run MVP tests** with 50-100 messages
3. ⭕ **Expand data** to 5000 Q&A pairs
4. ⭕ **Deploy to cloud** (AWS, Railway, Render)
5. ⭕ **Build frontend** (Next.js, React)
6. ⭕ **Scale to production**

---

## Resources

- **Docs**: http://localhost:8000/docs (Swagger UI)
- **Schema**: [schema.sql](schema.sql)
- **ML Pipeline**: [ml_pipeline.py](ml_pipeline.py)
- **Data Loader**: [data_loader.py](data_loader.py)
- **Configuration**: [config.py](config.py)
- **Data Requirements**: [DATA_REQUIREMENTS.py](DATA_REQUIREMENTS.py)

---

## Support & Contributions

For issues, questions, or contributions:

- Check existing issues on GitHub
- Review documentation in code
- Contact team: arshpurohit@gmail.com
