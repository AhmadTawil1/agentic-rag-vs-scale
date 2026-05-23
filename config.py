"""
Configuration file for the RAG research project.
"""

# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Free, local embeddings
# Alternative: "text-embedding-3-small" for OpenAI (requires API key)

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ChromaDB Configuration
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "rag_research_docs"

# Model Configuration
# System A (Goliath)
GOLIATH_MODEL = "gpt-5.4-mini"
GOLIATH_TEMPERATURE = 0.0

# System B (David)
DAVID_MODEL = "llama-3.2-1b"  # 1B parameter model (smallest)
DAVID_TEMPERATURE = 0.0
DAVID_API_BASE = "http://localhost:11434"  # Ollama default

# Agentic Workflow Configuration
MAX_RETRIEVAL_LOOPS = 3
MAX_GENERATION_RETRIES = 2
RETRIEVAL_TOP_K = 5

# Evaluation Configuration
RAGAS_METRICS = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]

# HotpotQA Benchmark
HOTPOTQA_COLLECTION_NAME = "hotpotqa_dev"
HOTPOTQA_CHROMA_DIR = "./chroma_hotpotqa"
HOTPOTQA_SAMPLE_SIZE = 200   # number of questions sampled from the validation split
HOTPOTQA_SEED = 42