"""
Modern ML Pipeline for Serenica
- Mistral 7B for text generation
- Sentence Transformers for embeddings
- LangChain for RAG
- Safety guardrails for crisis detection
"""

import logging
from typing import List, Dict, Optional, Tuple
import re
import asyncio
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_CONFIG = {
    "generation": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.1",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
    },
    "embeddings": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": 384,
    }
}

CRISIS_KEYWORDS = {
    "critical": [
        "suicide", "kill myself", "end my life", "not worth living",
        "harm myself", "cut myself", "overdose", "jump"
    ],
    "high": [
        "self harm", "hurt myself", "want to die", "suicidal thoughts",
        "severe depression", "panic attack", "anxiety attack"
    ],
    "medium": [
        "depressed", "hopeless", "anxious", "overwhelmed", "can't cope",
        "feeling low", "struggling"
    ]
}

THERAPY_TOPICS = [
    "anxiety", "depression", "relationship", "grief", "self-esteem",
    "work-life", "trauma", "addiction", "sleep", "stress"
]

# ============================================================
# EMBEDDINGS PROVIDER
# ============================================================

class EmbeddingsProvider:
    """Handles text embeddings using sentence-transformers"""
    
    def __init__(self):
        self.model = None
        
    async def initialize(self):
        """Load embeddings model"""
        try:
            model_name = MODEL_CONFIG["embeddings"]["model_name"]
            self.model = SentenceTransformer(model_name)
            logger.info(f"✓ Embeddings model loaded: {model_name}")
        except Exception as e:
            logger.error(f"✗ Failed to load embeddings: {e}")
            raise
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Embed single text"""
        if isinstance(text, list):
            return self.model.encode(text)
        return self.model.encode([text])[0]
    
    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts"""
        return self.model.encode(texts)
    
    async def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        embeddings = self.model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1])
        return float(similarity)

# ============================================================
# SAFETY GUARDRAILS
# ============================================================

class SafetyGuard:
    """Detects crisis situations and harmful content"""
    
    def __init__(self):
        self.embeddings = EmbeddingsProvider()
        
    async def initialize(self):
        """Initialize safety guardrails"""
        await self.embeddings.initialize()
        logger.info("✓ Safety guardrails initialized")
    
    async def detect_crisis(self, text: str) -> Tuple[bool, str, int]:
        """
        Detect crisis indicators in text
        Returns: (is_crisis, crisis_type, severity)
        """
        text_lower = text.lower()
        
        # Check critical keywords
        for keyword in CRISIS_KEYWORDS["critical"]:
            if keyword in text_lower:
                return True, "critical_crisis", 5
        
        # Check high severity
        for keyword in CRISIS_KEYWORDS["high"]:
            if keyword in text_lower:
                return True, "high_risk", 4
        
        # Check medium
        for keyword in CRISIS_KEYWORDS["medium"]:
            if keyword in text_lower:
                return True, "concerning", 2
        
        return False, "none", 0
    
    async def validate_bot_response(self, response: str) -> bool:
        """Ensure bot response is appropriate"""
        harmful_patterns = [
            r"kill yourself",
            r"hurt yourself",
            r"not helpful",
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, response.lower()):
                return False
        return True
    
    async def sanitize_input(self, text: str) -> str:
        """Sanitize user input"""
        # Remove HTML/injection attempts
        text = re.sub(r'<[^>]+>', '', text)
        # Remove SQL injection patterns
        text = re.sub(r'(--|;|\'|")', '', text)
        return text.strip()

# ============================================================
# TOPIC CLASSIFIER (Semantic Matching)
# ============================================================

class TopicClassifier:
    """Classifies user messages into therapy topics"""
    
    def __init__(self, embeddings: EmbeddingsProvider):
        self.embeddings = embeddings
        self.topic_descriptions = {
            "anxiety": "nervousness, worry, panic attacks, fear, stress",
            "depression": "sadness, hopelessness, low mood, emptiness",
            "relationship": "partner, family, friend, communication, conflict",
            "grief": "loss, death, mourning, sadness from death",
            "self-esteem": "confidence, worth, self-image, insecurity",
            "work-life": "job, career, work-life balance, productivity",
            "trauma": "past events, abuse, PTSD, triggers",
            "addiction": "substance use, behavioral addiction, dependency",
            "sleep": "insomnia, nightmares, sleep quality, fatigue",
            "stress": "overwhelm, pressure, burnout, coping"
        }
        self.topic_embeddings = None
    
    async def initialize(self):
        """Cache topic embeddings"""
        descriptions = list(self.topic_descriptions.values())
        self.topic_embeddings = await self.embeddings.embed_texts(descriptions)
        logger.info("✓ Topic classifier initialized")
    
    async def classify(self, text: str) -> Tuple[str, float]:
        """
        Classify text to topic with confidence
        Returns: (topic, confidence)
        """
        user_embedding = await self.embeddings.embed_text(text)
        
        # Calculate similarity to each topic
        similarities = np.dot(self.topic_embeddings, user_embedding)
        best_idx = np.argmax(similarities)
        confidence = float(similarities[best_idx])
        
        topic = list(self.topic_descriptions.keys())[best_idx]
        return topic, confidence

# ============================================================
# LANGUAGE GENERATION
# ============================================================

class TherapyGenerator:
    """Generates therapy responses using Mistral or Ollama"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.use_ollama = False
        
    async def initialize(self):
        """Initialize language model"""
        try:
            # Try local Ollama first
            try:
                import ollama
                self.use_ollama = True
                logger.info("✓ Using Ollama for inference")
                return
            except ImportError:
                pass
            
            # Fallback to local transformers (requires GPU for reasonable performance)
            model_name = MODEL_CONFIG["generation"]["model_name"]
            device = MODEL_CONFIG["generation"]["device"]
            
            if device == "cpu":
                logger.warning("⚠ Using CPU for generation (slow). Consider using Ollama or GPU.")
            
            logger.info(f"Loading model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=device,
            )
            logger.info("✓ Language model loaded")
        except Exception as e:
            logger.error(f"✗ Failed to initialize generator: {e}")
            raise
    
    async def generate(self, prompt: str, context: str = "") -> str:
        """Generate therapy response"""
        if self.use_ollama:
            return await self._generate_ollama(prompt, context)
        else:
            return await self._generate_transformers(prompt, context)
    
    async def _generate_ollama(self, prompt: str, context: str) -> str:
        """Generate using Ollama"""
        try:
            import ollama
            
            system_prompt = f"""You are Serenica, a compassionate AI therapist assistant.
            
Context: {context}

Guidelines:
- Be empathetic and non-judgmental
- Ask clarifying questions
- Provide coping strategies when appropriate
- If the user is in crisis, recommend professional help
- Keep responses concise (2-3 sentences)
"""
            
            response = ollama.generate(
                model="mistral",
                prompt=prompt,
                system=system_prompt,
                stream=False,
            )
            return response['response'].strip()
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise
    
    async def _generate_transformers(self, prompt: str, context: str) -> str:
        """Generate using local transformers"""
        full_prompt = f"""[INST] {prompt}

Context: {context}

Respond as a compassionate therapist assistant. Keep response concise. [/INST]"""
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=MODEL_CONFIG["generation"]["max_tokens"],
                temperature=MODEL_CONFIG["generation"]["temperature"],
                top_p=MODEL_CONFIG["generation"]["top_p"],
            )
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # Extract only the generated part (after [/INST])
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        
        return response

# ============================================================
# RAG ENGINE
# ============================================================

class RAGEngine:
    """Retrieval-Augmented Generation for knowledge base"""
    
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.rag_chain = None
        
    async def initialize(self):
        """Initialize RAG engine"""
        try:
            from langchain.document_loaders import CSVLoader
            from langchain.text_splitter import CharacterTextSplitter
            from langchain.vectorstores import FAISS
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=MODEL_CONFIG["embeddings"]["model_name"]
            )
            
            # Load knowledge base (will be populated from DB)
            logger.info("✓ RAG engine initialized (awaiting knowledge base)")
        except Exception as e:
            logger.error(f"✗ RAG initialization failed: {e}")
            raise
    
    async def index_knowledge_base(self, documents: List[Dict]):
        """Index knowledge base documents"""
        from langchain.schema import Document
        
        try:
            docs = [
                Document(
                    page_content=doc["answer"],
                    metadata={
                        "question": doc["question"],
                        "topic": doc.get("topic"),
                        "source": doc.get("source", "knowledge_base")
                    }
                )
                for doc in documents
            ]
            
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = text_splitter.split_documents(docs)
            
            self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
            logger.info(f"✓ Indexed {len(split_docs)} knowledge base documents")
        except Exception as e:
            logger.error(f"Failed to index knowledge base: {e}")
    
    async def retrieve_context(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant context for query"""
        if not self.vector_store:
            return []
        
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return []

# ============================================================
# MAIN ML PIPELINE
# ============================================================

class MLPipeline:
    """Main ML pipeline orchestrator"""
    
    def __init__(self):
        self.embeddings = EmbeddingsProvider()
        self.safety_guard = SafetyGuard()
        self.topic_classifier = None
        self.generator = TherapyGenerator()
        self.rag_engine = RAGEngine()
        
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing ML Pipeline...")
        
        await self.embeddings.initialize()
        await self.safety_guard.initialize()
        
        self.topic_classifier = TopicClassifier(self.embeddings)
        await self.topic_classifier.initialize()
        
        await self.generator.initialize()
        await self.rag_engine.initialize()
        
        logger.info("✓ ML Pipeline fully initialized")
    
    async def process_message(
        self,
        user_message: str,
        conversation_history: List[Dict],
        rag_context: List[str] = None
    ) -> Dict:
        """
        Main processing pipeline
        Returns: {response, topic, confidence, is_crisis, metadata}
        """
        try:
            # 1. Sanitize input
            clean_message = await self.safety_guard.sanitize_input(user_message)
            
            # 2. Detect crisis
            is_crisis, crisis_type, severity = await self.safety_guard.detect_crisis(clean_message)
            
            # 3. Classify topic
            topic, confidence = await self.topic_classifier.classify(clean_message)
            
            # 4. Retrieve RAG context
            if not rag_context:
                rag_context = await self.rag_engine.retrieve_context(clean_message)
            
            # 5. Build context for generation
            context = "\n".join(rag_context) if rag_context else ""
            
            # Build conversation context
            conv_context = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in conversation_history[-5:]  # Last 5 messages
            ])
            
            # 6. Generate response
            full_context = f"Topic: {topic}\nPrevious context:\n{conv_context}\n\nKnowledge:\n{context}"
            response = await self.generator.generate(clean_message, full_context)
            
            # 7. Validate response
            is_valid = await self.safety_guard.validate_bot_response(response)
            
            if not is_valid:
                response = "I appreciate you sharing that. Let me help you find better support."
            
            # 8. If crisis detected, add resources
            if is_crisis:
                response += "\n\n⚠️ I notice you might be in distress. Please reach out to a professional:\n- National Suicide Prevention Lifeline: 988\n- Crisis Text Line: Text HOME to 741741"
            
            return {
                "response": response,
                "topic": topic,
                "confidence": float(confidence),
                "is_crisis": is_crisis,
                "crisis_type": crisis_type,
                "severity": severity,
                "valid": is_valid,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "rag_sources": len(rag_context) if rag_context else 0,
                    "conv_length": len(conversation_history)
                }
            }
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise
