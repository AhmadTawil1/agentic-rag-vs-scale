"""
Tournament Runner: evaluates both RAG systems on the HotpotQA benchmark.

Prerequisites:
    python ingestion/ingest_hotpotqa.py   # run once to build the corpus

Then:
    python evaluation/run_tournament.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import (
    EMBEDDING_MODEL,
    HOTPOTQA_CHROMA_DIR,
    HOTPOTQA_COLLECTION_NAME,
    GOLIATH_MODEL,
)
from evaluation.load_benchmark import load_benchmark
from systems.system_a_goliath import run_baseline_rag
from systems.system_b_david import run_agentic_rag, create_agentic_rag

RESULTS_DIR = Path(__file__).parent / "results"


def run_tournament(questions: list[dict] | None = None) -> tuple[Path, Path]:
    if questions is None:
        questions = load_benchmark()

    print("=" * 80)
    print("RAG SYSTEMS TOURNAMENT  —  HotpotQA Benchmark")
    print("=" * 80)
    print(f"Questions : {len(questions)}")
    print(f"Collection: {HOTPOTQA_COLLECTION_NAME}  ({HOTPOTQA_CHROMA_DIR})")
    print(f"System A  : {GOLIATH_MODEL}  (naive RAG)")
    print(f"System B  : llama-3.1-8b-instant via Groq  (agentic RAG)")
    print("=" * 80)

    # --- Initialise shared resources once ---
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
    print("Compiling System B graph...")
    app_b = create_agentic_rag(vectorstore=vectorstore)
    print("Ready.\n")

    results_a: list[dict] = []
    results_b: list[dict] = []

    # --- System A ---
    print("System A  (Goliath — Baseline)")
    for item in tqdm(questions, desc="System A"):
        try:
            r = run_baseline_rag(
                item["question"],
                question_id=item["id"],
                vectorstore=vectorstore,
            )
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

    # --- System B ---
    print("\nSystem B  (David — Agentic)")
    for item in tqdm(questions, desc="System B"):
        try:
            r = run_agentic_rag(
                item["question"],
                question_id=item["id"],
                app=app_b,
            )
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

    # --- Persist ---
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_a = RESULTS_DIR / f"system_a_results_{timestamp}.json"
    path_b = RESULTS_DIR / f"system_b_results_{timestamp}.json"

    with open(path_a, "w", encoding="utf-8") as f:
        json.dump(results_a, f, indent=2, ensure_ascii=False)
    with open(path_b, "w", encoding="utf-8") as f:
        json.dump(results_b, f, indent=2, ensure_ascii=False)

    ok_a = sum(1 for r in results_a if not r["answer"].startswith("ERROR"))
    ok_b = sum(1 for r in results_b if not r["answer"].startswith("ERROR"))

    print("\n" + "=" * 80)
    print("TOURNAMENT COMPLETE")
    print("=" * 80)
    print(f"System A: {ok_a}/{len(questions)} completed  →  {path_a}")
    print(f"System B: {ok_b}/{len(questions)} completed  →  {path_b}")

    return path_a, path_b


if __name__ == "__main__":
    run_tournament()
