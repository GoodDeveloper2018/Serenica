"""
DATA REQUIREMENTS GUIDE FOR SERENICA
Comprehensive data strategy for therapy chatbot
"""

# ==============================================================
# DATA VOLUME SPECIFICATIONS
# ==============================================================

DATA_VOLUME = {
    "minimum_viable_product": {
        "qa_pairs": 2000,
        "conversation_examples": 300,
        "crisis_resources": 50,
        "expected_performance": "Basic functionality, ~70% response quality",
        "training_time": "1-2 weeks",
        "cost_estimate": "Low (existing datasets)"
    },
    "production_baseline": {
        "qa_pairs": 5000,
        "conversation_examples": 1000,
        "crisis_resources": 200,
        "expected_performance": "Good, ~85% response quality",
        "training_time": "2-4 weeks",
        "cost_estimate": "Medium (licensed + custom)"
    },
    "optimal_large_scale": {
        "qa_pairs": 10000,
        "conversation_examples": 5000,
        "crisis_resources": 500,
        "expected_performance": "Excellent, 90%+ response quality",
        "training_time": "4-8 weeks",
        "cost_estimate": "High (comprehensive collection)"
    },
    "enterprise_gold_standard": {
        "qa_pairs": 50000,
        "conversation_examples": 20000,
        "crisis_resources": 2000,
        "expected_performance": "Outstanding, 95%+ response quality",
        "training_time": "3-6 months",
        "cost_estimate": "Very High (enterprise partnerships)"
    }
}

# ==============================================================
# DATA COLLECTION STRATEGIES
# ==============================================================

DATA_COLLECTION_STRATEGIES = {
    "existing_datasets": {
        "CounselChat": {
            "url": "https://counselchat.com",
            "coverage": "~2000 Q&A pairs",
            "quality": "Therapist-verified",
            "cost": "Free/open-source",
            "topics": ["anxiety", "depression", "relationships", "career"],
            "notes": "Good starting point, already included in project"
        },
        "Mental Health America": {
            "url": "https://www.mhanational.org",
            "coverage": "Resource database",
            "quality": "Professional",
            "cost": "Free",
            "topics": ["depression", "anxiety", "mental health"],
            "notes": "Great for crisis resources"
        },
        "SAMHSA": {
            "url": "https://www.samhsa.gov",
            "coverage": "Treatment resources, publications",
            "quality": "Government-verified",
            "cost": "Free",
            "topics": ["Substance abuse", "mental health treatment"],
            "notes": "Authoritative source"
        },
        "Psychology Today": {
            "url": "https://www.psychologytoday.com",
            "coverage": "Therapist directory, articles",
            "quality": "Professional",
            "cost": "$$ (requires API/partnership)",
            "topics": ["All therapy topics"],
            "notes": "Requires partnership agreement"
        }
    },
    
    "community_collection": {
        "therapist_partnerships": {
            "description": "Partner with therapist networks",
            "implementation": [
                "Contact therapist associations (NASW, AAMFT, APA)",
                "Offer data sharing incentives",
                "Implement review workflow",
                "Verify credentials"
            ],
            "expected_qa_pairs": 2000,
            "quality": "High",
            "timeline": "2-3 months",
            "cost": "Low-Medium (incentives)"
        },
        "user_feedback_loop": {
            "description": "Collect from user interactions",
            "implementation": [
                "Track user ratings (1-5 stars)",
                "Collect 'helpful'/'not helpful' votes",
                "Implement feedback forms",
                "Monthly review cycles"
            ],
            "expected_qa_pairs": "Continuous growth",
            "quality": "Improves over time",
            "timeline": "Ongoing",
            "cost": "Minimal (part of operations)"
        },
        "crowdsourcing": {
            "description": "Amazon Mechanical Turk, Figure Eight",
            "implementation": [
                "Create HIT templates",
                "Screen for mental health knowledge",
                "Multi-reviewer validation",
                "Quality control metrics"
            ],
            "expected_qa_pairs": 3000,
            "quality": "Medium (needs review)",
            "timeline": "4-8 weeks",
            "cost": "$5,000-$15,000"
        }
    },
    
    "proprietary_development": {
        "manual_creation": {
            "description": "Hire therapists to create content",
            "implementation": [
                "Hire 3-5 contract therapists",
                "Define quality guidelines",
                "Weekly review meetings",
                "Iterative refinement"
            ],
            "expected_qa_pairs": 1000,
            "quality": "Excellent",
            "timeline": "3 months",
            "cost": "$30,000-$50,000"
        },
        "fine_tuning_conversations": {
            "description": "Anonymized therapy conversation data",
            "implementation": [
                "HIPAA-compliant data collection",
                "Patient consent (opt-in)",
                "Professional de-identification",
                "Regulatory review"
            ],
            "expected_qa_pairs": "N/A (continuous)",
            "quality": "Very High",
            "timeline": "Ongoing + 3 month setup",
            "cost": "Medium (compliance infrastructure)"
        }
    }
}

# ==============================================================
# DATA QUALITY STANDARDS
# ==============================================================

QUALITY_STANDARDS = {
    "content_requirements": {
        "medical_accuracy": {
            "requirement": "All medical/psychological claims must be evidence-based",
            "verification": "Cross-reference with peer-reviewed research",
            "review_by": "Licensed therapist",
            "score": "Must pass validation"
        },
        "ethical_alignment": {
            "requirement": "Align with therapeutic best practices",
            "verification": "Ethics review by professional",
            "review_by": "LMSW, LCSW, or Licensed Psychologist",
            "score": "Must meet 80%+ on ethics rubric"
        },
        "diversity_inclusion": {
            "requirement": "Culturally sensitive, diverse scenarios",
            "verification": "Review by diverse cultural consultants",
            "review_by": "Cultural competency expert",
            "score": "Must cover 5+ cultural backgrounds"
        },
        "clarity": {
            "requirement": "Language suitable for general public",
            "verification": "Readability score (Flesch-Kincaid grade 8-10)",
            "review_by": "Content editor",
            "score": "Must achieve target grade level"
        }
    },
    
    "crisis_safety": {
        "suicide_prevention": {
            "requirement": "Responses never encourage self-harm",
            "verification": "Automated flagging + manual review",
            "review_by": "Crisis specialist",
            "score": "100% compliance required"
        },
        "resource_accuracy": {
            "requirement": "All crisis resources must be current/verified",
            "verification": "Quarterly updates, hotline testing",
            "review_by": "Crisis coordinator",
            "score": "100% accuracy"
        },
        "escalation_protocol": {
            "requirement": "System knows when to suggest professional help",
            "verification": "Test with crisis indicators",
            "review_by": "Clinical supervisor",
            "score": "100% detection required"
        }
    },
    
    "diversity_metrics": {
        "topic_distribution": {
            "anxiety": "15-20%",
            "depression": "15-20%",
            "relationships": "15-20%",
            "grief": "10-15%",
            "self-esteem": "10-15%",
            "work-life": "10-15%",
            "other": "5-10%"
        },
        "cultural_backgrounds": [
            "Western/European",
            "Asian",
            "Latino/Hispanic",
            "African American",
            "Middle Eastern",
            "LGBTQ+",
            "Religious minorities",
            "Socioeconomic diversity"
        ],
        "language_coverage": [
            "English (priority)",
            "Spanish",
            "Mandarin",
            "French",
            "German"
        ]
    }
}

# ==============================================================
# DATA COLLECTION PIPELINE
# ==============================================================

COLLECTION_PIPELINE = {
    "phase_1_bootstrap": {
        "name": "Initial Data Collection (Weeks 1-4)",
        "tasks": [
            "Download and process CounselChat dataset",
            "Contact SAMHSA for resource data",
            "Scrape Psychology Today articles (with permission)",
            "Create 500 custom Q&A pairs"
        ],
        "target_volume": 2000,
        "resource_requirements": "1 data engineer, 1 therapist reviewer"
    },
    
    "phase_2_validation": {
        "name": "Quality Review & Curation (Weeks 3-6)",
        "tasks": [
            "Therapist review all content",
            "Ethics review",
            "Remove duplicates",
            "Create embeddings",
            "Set up RAG index"
        ],
        "target_volume": 1500-1800 (after filtering),
        "resource_requirements": "2 therapists, 1 QA engineer"
    },
    
    "phase_3_expansion": {
        "name": "Scale to Production Volume (Weeks 7-14)",
        "tasks": [
            "Launch crowdsourcing campaign",
            "Partner with therapist network",
            "Implement feedback loop",
            "Monthly data refresh"
        ],
        "target_volume": 5000+,
        "resource_requirements": "1 data manager, 2 therapists, 1 compliance officer"
    },
    
    "phase_4_ongoing": {
        "name": "Continuous Improvement (Ongoing)",
        "tasks": [
            "Monthly quality audits",
            "Quarterly crisis resource updates",
            "User feedback incorporation",
            "Model performance monitoring"
        ],
        "target_volume": "Grow by 10-15% monthly",
        "resource_requirements": "1 data manager, 1 QA specialist"
    }
}

# ==============================================================
# IMMEDIATE NEXT STEPS
# ==============================================================

IMMEDIATE_ACTIONS = """
1. DATA PREPARATION (This Week)
   - Extract counselchat-data.csv properly formatted
   - Create 300-500 custom Q&A pairs (high quality examples)
   - Verify against QUALITY_STANDARDS
   
2. INFRASTRUCTURE SETUP
   - Deploy PostgreSQL with pgvector
   - Set up embeddings pipeline
   - Initialize RAG index
   - Run schema.sql
   
3. DATA INGESTION
   - Use DataImporter class to load initial dataset
   - Validate all records
   - Generate embeddings
   - Test retrieval (RAG)
   
4. QUALITY ASSURANCE
   - Manual review of 10% sample
   - Test crisis detection
   - Validate response quality
   - Run end-to-end tests

5. SCALE PLANNING
   - Identify therapist partners
   - Define crowdsourcing strategy
   - Set up feedback collection UI
   - Plan monthly expansion cycles

EXPECTED TIMELINE:
- MVP (2000 QA pairs): 2-3 weeks
- Production (5000 QA pairs): 6-8 weeks  
- Optimal (10000+ QA pairs): 3-4 months
"""

# ==============================================================
# DATA FORMAT SPECIFICATION
# ==============================================================

DATA_SCHEMA_DETAILS = {
    "CSV_Format": {
        "columns": {
            "question": {
                "type": "string",
                "required": True,
                "min_length": 5,
                "max_length": 500,
                "example": "How do I manage anxiety during presentations?",
                "validation": "Non-empty, realistic therapy question"
            },
            "answer": {
                "type": "string",
                "required": True,
                "min_length": 20,
                "max_length": 2000,
                "example": "Anxiety during presentations is common. Try: 1) Deep breathing exercises...",
                "validation": "Evidence-based, actionable advice"
            },
            "topic": {
                "type": "enum",
                "required": True,
                "allowed": ["anxiety", "depression", "relationship", "grief", "self-esteem", "work-life", "trauma", "addiction", "sleep", "stress"],
                "example": "anxiety",
                "validation": "Must match allowed topic"
            },
            "source": {
                "type": "string",
                "required": False,
                "example": "CounselChat",
                "validation": "Attribution source"
            },
            "therapist_name": {
                "type": "string",
                "required": False,
                "example": "Dr. Jane Smith, LMFT",
                "validation": "Professional credentials if provided"
            },
            "therapist_verified": {
                "type": "boolean",
                "required": False,
                "default": False,
                "validation": "Only True if verified by licensed therapist"
            }
        },
        "example_rows": [
            {
                "question": "I keep having panic attacks at work",
                "answer": "Panic attacks at work are manageable. Progressive muscle relaxation and box breathing (4-4-4-4) help...",
                "topic": "anxiety",
                "source": "Clinical Experience",
                "therapist_name": "Dr. John Doe, LCSW",
                "therapist_verified": True
            },
            {
                "question": "My relationship feels distant",
                "answer": "Distance in relationships often stems from communication gaps. Try scheduling regular conversations...",
                "topic": "relationship",
                "source": "Research",
                "therapist_verified": False
            }
        ]
    }
}

# ==============================================================
# COST-BENEFIT ANALYSIS
# ==============================================================

COST_BENEFIT = """
MINIMUM VIABLE PRODUCT (2,000 Q&A pairs)
- Time Investment: 2-3 weeks
- Cost: $0-$5,000 (using free datasets + in-house)
- Expected Response Quality: 70%
- Deployment: Proof of concept, hackathon level

PRODUCTION (5,000 Q&A pairs)
- Time Investment: 6-8 weeks
- Cost: $10,000-$30,000
- Expected Response Quality: 85%
- Deployment: Ready for beta launch

OPTIMAL (10,000+ Q&A pairs)  
- Time Investment: 3-4 months
- Cost: $50,000-$100,000
- Expected Response Quality: 90%+
- Deployment: Production-ready, differentiating

ROI PROJECTION:
- Each 1000 Q&A pairs: +5% improvement in response quality
- User satisfaction increases ~8% per quality point
- Retention improves ~12% with better response quality
- Cost per user acquisition becomes 20% lower with good quality
"""

if __name__ == "__main__":
    print(IMMEDIATE_ACTIONS)
    print("\n" + "="*60 + "\n")
    print(COST_BENEFIT)
