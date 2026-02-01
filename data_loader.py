"""
Data Interface & Loader for Serenica
Handles ingestion, validation, and embedding of therapy data
"""

import logging
import csv
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import hashlib

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update

logger = logging.getLogger(__name__)

# ============================================================
# DATA MODELS
# ============================================================

class TherapyDataValidator:
    """Validates therapy dataset for quality"""
    
    REQUIRED_FIELDS = ["question", "answer", "topic"]
    ALLOWED_TOPICS = [
        "anxiety", "depression", "relationship", "grief", "self-esteem",
        "work-life", "trauma", "addiction", "sleep", "stress"
    ]
    
    @staticmethod
    def validate_row(row: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a single row
        Returns: (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        for field in TherapyDataValidator.REQUIRED_FIELDS:
            if field not in row or not row[field]:
                errors.append(f"Missing required field: {field}")
        
        # Validate topic
        if "topic" in row and row["topic"] not in TherapyDataValidator.ALLOWED_TOPICS:
            errors.append(f"Invalid topic: {row['topic']}. Must be one of {TherapyDataValidator.ALLOWED_TOPICS}")
        
        # Validate text length
        if "question" in row and len(row["question"].strip()) < 5:
            errors.append("Question too short (min 5 characters)")
        
        if "answer" in row and len(row["answer"].strip()) < 20:
            errors.append("Answer too short (min 20 characters)")
        
        # Check for HTML/injection
        for field in ["question", "answer"]:
            if field in row and ("<" in row[field] or ";" in row[field]):
                errors.append(f"Field '{field}' contains potentially malicious content")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text for processing"""
        # Remove extra whitespace
        text = " ".join(text.split())
        # Remove quotes at start/end
        text = text.strip('"\'')
        return text

# ============================================================
# CSV DATA LOADER
# ============================================================

class CSVDataLoader:
    """Loads and processes CSV files"""
    
    def __init__(self):
        self.validator = TherapyDataValidator()
    
    async def load_csv(
        self,
        file_path: str,
        skip_validation: bool = False
    ) -> Tuple[List[Dict], Dict]:
        """
        Load CSV file
        Returns: (valid_rows, statistics)
        """
        try:
            df = pd.read_csv(file_path)
            valid_rows = []
            stats = {
                "total_rows": len(df),
                "valid_rows": 0,
                "invalid_rows": 0,
                "topics": {},
                "errors": []
            }
            
            for idx, row in df.iterrows():
                row_dict = row.to_dict()
                
                # Clean text
                for field in ["question", "answer"]:
                    if field in row_dict:
                        row_dict[field] = self.validator.clean_text(str(row_dict[field]))
                
                # Validate
                if not skip_validation:
                    is_valid, errors = self.validator.validate_row(row_dict)
                    if not is_valid:
                        stats["errors"].append({
                            "row": idx + 2,  # Excel row number
                            "errors": errors
                        })
                        stats["invalid_rows"] += 1
                        continue
                
                # Count topic
                topic = row_dict.get("topic", "unknown")
                stats["topics"][topic] = stats["topics"].get(topic, 0) + 1
                
                valid_rows.append(row_dict)
                stats["valid_rows"] += 1
            
            logger.info(f"✓ Loaded {stats['valid_rows']}/{stats['total_rows']} valid rows from {file_path}")
            return valid_rows, stats
        
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise

# ============================================================
# DATABASE LOADER
# ============================================================

class DatabaseLoader:
    """Loads data into PostgreSQL with embeddings"""
    
    def __init__(self, ml_pipeline=None):
        """
        Initialize loader
        ml_pipeline: MLPipeline instance for generating embeddings
        """
        self.ml_pipeline = ml_pipeline
    
    async def load_knowledge_base(
        self,
        rows: List[Dict],
        db: AsyncSession
    ) -> Dict:
        """
        Load therapy Q&A into knowledge base
        Returns: loading statistics
        """
        from app.models import TherapyKnowledgeBase
        
        stats = {
            "loaded": 0,
            "failed": 0,
            "errors": []
        }
        
        for row in rows:
            try:
                # Generate embedding if available
                embedding = None
                if self.ml_pipeline:
                    embedding_vec = await self.ml_pipeline.embeddings.embed_text(
                        row.get("question", "") + " " + row.get("answer", "")
                    )
                    embedding = embedding_vec.tolist()  # Convert to list for storage
                
                # Create knowledge base record
                kb_entry = TherapyKnowledgeBase(
                    question=row.get("question"),
                    answer=row.get("answer"),
                    topic=row.get("topic"),
                    source=row.get("source", "uploaded_data"),
                    therapist_name=row.get("therapist_name"),
                    therapist_verified=row.get("verified", False),
                    embedding=embedding,
                )
                
                db.add(kb_entry)
                stats["loaded"] += 1
            
            except Exception as e:
                logger.error(f"Failed to load row: {e}")
                stats["failed"] += 1
                stats["errors"].append(str(e))
        
        try:
            await db.commit()
            logger.info(f"✓ Loaded {stats['loaded']} knowledge base entries")
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to commit: {e}")
            stats["failed"] += len(rows) - stats["loaded"]
        
        return stats

# ============================================================
# CRISIS RESOURCES LOADER
# ============================================================

class CrisisResourcesLoader:
    """Loads crisis resources into database"""
    
    # Default crisis resources
    DEFAULT_RESOURCES = [
        {
            "resource_name": "National Suicide Prevention Lifeline",
            "resource_type": "hotline",
            "phone_number": "988",
            "country": "USA",
            "description": "24/7 free and confidential support for people in distress"
        },
        {
            "resource_name": "Crisis Text Line",
            "resource_type": "text_service",
            "description": "Text HOME to 741741 for 24/7 crisis support",
            "country": "USA"
        },
        {
            "resource_name": "International Association for Suicide Prevention",
            "resource_type": "website",
            "website_url": "https://www.iasp.info/resources/Crisis_Centres/",
            "description": "Global directory of crisis centers",
            "country": "Global"
        },
        {
            "resource_name": "SAMHSA National Helpline",
            "resource_type": "hotline",
            "phone_number": "1-800-662-4357",
            "country": "USA",
            "description": "Free, confidential substance abuse and mental health treatment referral"
        },
        {
            "resource_name": "MIND UK",
            "resource_type": "website",
            "website_url": "https://www.mind.org.uk/",
            "country": "UK",
            "description": "Mental health information and support"
        }
    ]
    
    @staticmethod
    async def load_default_resources(db: AsyncSession) -> Dict:
        """Load default crisis resources"""
        from app.models import CrisisResource
        
        stats = {"loaded": 0, "failed": 0}
        
        for resource in CrisisResourcesLoader.DEFAULT_RESOURCES:
            try:
                # Check if already exists
                existing = await db.execute(
                    select(CrisisResource).where(
                        CrisisResource.resource_name == resource["resource_name"]
                    )
                )
                if existing.scalars().first():
                    continue
                
                cr = CrisisResource(**resource)
                db.add(cr)
                stats["loaded"] += 1
            except Exception as e:
                logger.error(f"Failed to load resource: {e}")
                stats["failed"] += 1
        
        await db.commit()
        logger.info(f"✓ Loaded {stats['loaded']} crisis resources")
        return stats

# ============================================================
# DATA SCHEMA / INTERFACE
# ============================================================

class DataSchema:
    """Defines expected data format"""
    
    @staticmethod
    def get_expected_schema() -> Dict:
        """Get expected CSV schema"""
        return {
            "question": {
                "type": "string",
                "required": True,
                "description": "Therapy question from user",
                "example": "How do I manage my anxiety?"
            },
            "answer": {
                "type": "string",
                "required": True,
                "description": "Therapist response",
                "example": "There are several evidence-based approaches..."
            },
            "topic": {
                "type": "string",
                "required": True,
                "description": "Category of question",
                "example": "anxiety",
                "allowed_values": [
                    "anxiety", "depression", "relationship", "grief",
                    "self-esteem", "work-life", "trauma", "addiction",
                    "sleep", "stress"
                ]
            },
            "source": {
                "type": "string",
                "required": False,
                "description": "Source of data",
                "example": "CounselChat, Clinical Research, etc."
            },
            "therapist_name": {
                "type": "string",
                "required": False,
                "description": "Therapist who provided the answer",
                "example": "Dr. Jane Smith, LMFT"
            },
            "therapist_verified": {
                "type": "boolean",
                "required": False,
                "description": "Whether therapist verified the answer",
                "example": True
            },
            "upvotes": {
                "type": "integer",
                "required": False,
                "description": "Community validation score",
                "example": 45
            }
        }
    
    @staticmethod
    def get_sample_csv() -> str:
        """Generate sample CSV"""
        return """question,answer,topic,source,therapist_name,therapist_verified
"How do I cope with anxiety when it strikes?","Anxiety management involves several techniques: 1) Box breathing (4-4-4-4), 2) Progressive muscle relaxation, 3) Grounding techniques. Practice these daily.",anxiety,Research,"Dr. Jane Smith, LMFT",true
"My relationship feels distant","Consider having an honest conversation about your feelings. Ask open-ended questions and listen actively. Consider couples therapy if issues persist.",relationship,CounselChat,"Dr. John Doe, LCSW",true
"I feel overwhelmed at work","Burnout is common. Try: setting boundaries, taking breaks, delegating tasks, and maintaining work-life balance. Short daily walks help tremendously.",work-life,Clinical Practice,"Dr. Sarah Johnson, LMFT",true"""

# ============================================================
# DATA IMPORTER (HIGH-LEVEL API)
# ============================================================

class DataImporter:
    """High-level interface for importing therapy data"""
    
    def __init__(self, ml_pipeline=None):
        self.csv_loader = CSVDataLoader()
        self.db_loader = DatabaseLoader(ml_pipeline)
        self.validator = TherapyDataValidator()
    
    async def import_csv(
        self,
        file_path: str,
        db: AsyncSession,
        skip_validation: bool = False
    ) -> Dict:
        """
        Import CSV file into database
        Returns: comprehensive import statistics
        """
        logger.info(f"Starting import from {file_path}")
        
        # Load and validate
        rows, csv_stats = await self.csv_loader.load_csv(file_path, skip_validation)
        
        # Load into database
        db_stats = await self.db_loader.load_knowledge_base(rows, db)
        
        # Load default crisis resources
        resources_stats = await CrisisResourcesLoader.load_default_resources(db)
        
        # Combine stats
        combined_stats = {
            "timestamp": datetime.now().isoformat(),
            "csv_stats": csv_stats,
            "database_stats": db_stats,
            "resources_stats": resources_stats,
            "summary": {
                "total_imported": db_stats["loaded"],
                "total_failed": db_stats["failed"],
                "resources_loaded": resources_stats["loaded"]
            }
        }
        
        logger.info(f"✓ Import complete: {db_stats['loaded']} records loaded")
        return combined_stats
    
    @staticmethod
    def get_import_template() -> Dict:
        """Get CSV template and schema"""
        return {
            "schema": DataSchema.get_expected_schema(),
            "sample": DataSchema.get_sample_csv(),
            "instructions": """
1. Create CSV with columns: question, answer, topic, source, therapist_name, therapist_verified
2. Use allowed topics: anxiety, depression, relationship, grief, self-esteem, work-life, trauma, addiction, sleep, stress
3. Ensure questions and answers are substantial (min 5 and 20 chars respectively)
4. Remove HTML/SQL injection patterns
5. Upload via /api/admin/import endpoint
            """
        }

# ============================================================
# BATCH PROCESSING
# ============================================================

class BatchProcessor:
    """Process data in batches for efficiency"""
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
    
    async def process_batches(
        self,
        rows: List[Dict],
        processor_fn,
        db: AsyncSession
    ) -> List[Dict]:
        """Process rows in batches"""
        results = []
        
        for i in range(0, len(rows), self.batch_size):
            batch = rows[i:i + self.batch_size]
            batch_result = await processor_fn(batch, db)
            results.append(batch_result)
            logger.info(f"✓ Processed batch {i // self.batch_size + 1}")
        
        return results
