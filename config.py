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
# System A (Goliath) — Naive RAG + GPT-5.4-mini
GOLIATH_MODEL = "gpt-5.4-mini"
GOLIATH_TEMPERATURE = 0.0

# System B (David) — Agentic RAG + Llama-3.1-8B via Groq
DAVID_MODEL = "llama-3.1-8b-instant"
DAVID_TEMPERATURE = 0.0

# System C (Hermes) — Naive RAG + Llama-3.1-8B via Groq
HERMES_MODEL = "llama-3.1-8b-instant"
HERMES_TEMPERATURE = 0.0

# Local Ollama fallback (same model, different name format used by Ollama)
OLLAMA_LLAMA_MODEL = "llama3.1:8b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Local GPU inference (Colab T4 / CUDA)
# Override via LOCAL_LLM_MODEL env var if needed
LOCAL_LLM_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# System D (Titan) — Agentic RAG + GPT-5.4-mini
TITAN_MODEL = "gpt-5.4-mini"
TITAN_TEMPERATURE = 0.0

# Evaluator
EVALUATOR_MODEL = "gpt-5.4"

# Agentic Workflow Configuration
MAX_RETRIEVAL_LOOPS = 3
MAX_GENERATION_RETRIES = 2
RETRIEVAL_TOP_K = 5

# Evaluation Configuration
RAGAS_METRICS = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]

# HotpotQA Benchmark
HOTPOTQA_COLLECTION_NAME = "hotpotqa_dev"
HOTPOTQA_CHROMA_DIR = "./chroma_hotpotqa"
HOTPOTQA_SAMPLE_SIZE = 100   # number of questions sampled from the validation split
HOTPOTQA_SEED = 42