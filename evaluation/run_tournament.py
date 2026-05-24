"""
Tournament Runner: evaluates all four RAG systems on the HotpotQA benchmark.

2x2 design:
    System A (Goliath) — Naive RAG     + GPT-5.4-mini
    System B (David)   — Agentic RAG   + Llama-3.1-8B  (Groq, with fallback)
    System C (Hermes)  — Naive RAG     + Llama-3.1-8B  (Groq, with fallback)
    System D (Titan)   — Agentic RAG   + GPT-5.4-mini

Prerequisites:
    python ingestion/ingest_hotpotqa.py   # run once to build the corpus

Then:
    python evaluation/run_tournament.py
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import (
    EMBEDDING_MODEL,
    GOLIATH_MODEL,
    HERMES_MODEL,
    HOTPOTQA_CHROMA_DIR,
    HOTPOTQA_COLLECTION_NAME,
    TITAN_MODEL,
)
from evaluation.load_benchmark import load_benchmark
from systems.system_a_goliath import run_baseline_rag
from systems.system_b_david import run_agentic_rag, create_agentic_rag
from systems.system_c_hermes import run_naive_llama_rag
from systems.system_d_titan import run_agentic_gpt_rag, create_agentic_gpt_rag

RESULTS_DIR = Path(__file__).parent / "results"


def run_tournament(questions: list[dict] | None = None) -> tuple[Path, Path, Path, Path]:
    if questions is None:
        questions = load_benchmark()

    print("=" * 80)
    print("RAG SYSTEMS TOURNAMENT  —  HotpotQA Benchmark  (2×2 Design)")
    print("=" * 80)
    print(f"Questions : {len(questions)}")
    print(f"Collection: {HOTPOTQA_COLLECTION_NAME}  ({HOTPOTQA_CHROMA_DIR})")
    llama_backend = (
        "Ollama (local GPU)" if os.getenv("USE_OLLAMA", "").lower() == "true"
        else "HuggingFace (local GPU)" if os.getenv("USE_LOCAL_LLM", "").lower() == "true"
        else "Groq API"
    )
    print(f"System A  : {GOLIATH_MODEL}  (naive RAG)")
    print(f"System B  : {HERMES_MODEL} via {llama_backend}  (agentic RAG)")
    print(f"System C  : {HERMES_MODEL} via {llama_backend}  (naive RAG)")
    print(f"System D  : {TITAN_MODEL}  (agentic RAG)")
    print("=" * 80)

    # --- Shared resources ---
    print("\nLoading embedding model and vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        persist_directory=HOTPOTQA_CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=HOTPOTQA_COLLECTION_NAME,
    )
    print("Compiling System B graph (David)...")
    app_b = create_agentic_rag(vectorstore=vectorstore)
    print("Compiling System D graph (Titan)...")
    app_d = create_agentic_gpt_rag(vectorstore=vectorstore)
    print("Ready.\n")

    results_a: list[dict] = []
    results_b: list[dict] = []
    results_c: list[dict] = []
    results_d: list[dict] = []

    # --- System A: Goliath (Naive GPT) ---
    print("System A  (Goliath — Naive GPT)")
    for item in tqdm(questions, desc="System A"):
        try:
            r = run_baseline_rag(item["question"], question_id=item["id"], vectorstore=vectorstore)
            results_a.append({
                "id": item["id"],
                "question": item["question"],
                "answer": r["answer"],
                "ground_truth": item["ground_truth"],
                "contexts": [d.page_content for d in r["source_documents"]],
                "type": item["type"],
                "level": item["level"],
            })
        except Exception as exc:
            results_a.append({
                "id": item["id"],
                "question": item["question"],
                "answer": f"ERROR: {exc}",
                "ground_truth": item["ground_truth"],
                "contexts": [],
                "type": item["type"],
                "level": item["level"],
            })

    # --- System B: David (Agentic Llama) ---
    print("\nSystem B  (David — Agentic Llama)")
    for item in tqdm(questions, desc="System B"):
        try:
            r = run_agentic_rag(item["question"], question_id=item["id"], app=app_b)
            results_b.append({
                "id": item["id"],
                "question": item["question"],
                "answer": r["answer"],
                "ground_truth": item["ground_truth"],
                "contexts": [d.page_content for d in r["documents"]],
                "retrieval_attempts": r["retrieval_attempts"],
                "generation_attempts": r["generation_attempts"],
                "type": item["type"],
                "level": item["level"],
            })
        except Exception as exc:
            results_b.append({
                "id": item["id"],
                "question": item["question"],
                "answer": f"ERROR: {exc}",
                "ground_truth": item["ground_truth"],
                "contexts": [],
                "retrieval_attempts": 0,
                "generation_attempts": 0,
                "type": item["type"],
                "level": item["level"],
            })

    # --- System C: Hermes (Naive Llama) ---
    print("\nSystem C  (Hermes — Naive Llama)")
    for item in tqdm(questions, desc="System C"):
        try:
            r = run_naive_llama_rag(item["question"], question_id=item["id"], vectorstore=vectorstore)
            results_c.append({
                "id": item["id"],
                "question": item["question"],
                "answer": r["answer"],
                "ground_truth": item["ground_truth"],
                "contexts": [d.page_content for d in r["source_documents"]],
                "type": item["type"],
                "level": item["level"],
            })
        except Exception as exc:
            results_c.append({
                "id": item["id"],
                "question": item["question"],
                "answer": f"ERROR: {exc}",
                "ground_truth": item["ground_truth"],
                "contexts": [],
                "type": item["type"],
                "level": item["level"],
            })

    # --- System D: Titan (Agentic GPT) ---
    print("\nSystem D  (Titan — Agentic GPT)")
    for item in tqdm(questions, desc="System D"):
        try:
            r = run_agentic_gpt_rag(item["question"], question_id=item["id"], app=app_d)
            results_d.append({
                "id": item["id"],
                "question": item["question"],
                "answer": r["answer"],
                "ground_truth": item["ground_truth"],
                "contexts": [d.page_content for d in r["documents"]],
                "retrieval_attempts": r["retrieval_attempts"],
                "generation_attempts": r["generation_attempts"],
                "type": item["type"],
                "level": item["level"],
            })
        except Exception as exc:
            results_d.append({
                "id": item["id"],
                "question": item["question"],
                "answer": f"ERROR: {exc}",
                "ground_truth": item["ground_truth"],
                "contexts": [],
                "retrieval_attempts": 0,
                "generation_attempts": 0,
                "type": item["type"],
                "level": item["level"],
            })

    # --- Persist ---
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_a = RESULTS_DIR / f"system_a_results_{timestamp}.json"
    path_b = RESULTS_DIR / f"system_b_results_{timestamp}.json"
    path_c = RESULTS_DIR / f"system_c_results_{timestamp}.json"
    path_d = RESULTS_DIR / f"system_d_results_{timestamp}.json"

    for path, results in [(path_a, results_a), (path_b, results_b), (path_c, results_c), (path_d, results_d)]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    ok_a = sum(1 for r in results_a if not r["answer"].startswith("ERROR"))
    ok_b = sum(1 for r in results_b if not r["answer"].startswith("ERROR"))
    ok_c = sum(1 for r in results_c if not r["answer"].startswith("ERROR"))
    ok_d = sum(1 for r in results_d if not r["answer"].startswith("ERROR"))
    n = len(questions)

    print("\n" + "=" * 80)
    print("TOURNAMENT COMPLETE")
    print("=" * 80)
    print(f"System A (Goliath): {ok_a}/{n} completed  →  {path_a}")
    print(f"System B (David)  : {ok_b}/{n} completed  →  {path_b}")
    print(f"System C (Hermes) : {ok_c}/{n} completed  →  {path_c}")
    print(f"System D (Titan)  : {ok_d}/{n} completed  →  {path_d}")

    return path_a, path_b, path_c, path_d


if __name__ == "__main__":
    run_tournament()
