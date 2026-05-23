"""
HotpotQA Ingestion Script

Downloads the HotpotQA distractor validation split, samples N questions, and ingests
all associated Wikipedia paragraphs into ChromaDB.  Each paragraph is tagged with
its question_id so retrieval can be scoped per question at evaluation time.

Run once before the tournament:
    python ingestion/ingest_hotpotqa.py
"""

import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from datasets import load_dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

from config import (
    EMBEDDING_MODEL,
    HOTPOTQA_CHROMA_DIR,
    HOTPOTQA_COLLECTION_NAME,
    HOTPOTQA_SAMPLE_SIZE,
    HOTPOTQA_SEED,
)

BENCHMARK_PATH = Path(__file__).parent.parent / "evaluation" / "benchmark_questions.json"


def build_corpus(n_samples: int = HOTPOTQA_SAMPLE_SIZE, seed: int = HOTPOTQA_SEED):
    """
    Sample questions from HotpotQA and ingest their context paragraphs into ChromaDB.

    HotpotQA distractor config provides 10 Wikipedia paragraphs per question
    (2 gold supporting + 8 distractors).  This mirrors a realistic retrieval scenario
    where the system must identify the relevant passages from a noisy candidate set.
    """
    print("Loading HotpotQA distractor split from HuggingFace...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    print(f"  Full validation set: {len(dataset)} questions")

    random.seed(seed)
    indices = random.sample(range(len(dataset)), n_samples)
    print(f"  Sampled {n_samples} questions (seed={seed})")

    documents: list[Document] = []
    questions: list[dict] = []

    for idx in tqdm(indices, desc="Building corpus"):
        sample = dataset[idx]
        qid = sample["id"]
        supporting_titles = set(sample["supporting_facts"]["title"])

        # Each context entry is [title, [sent0, sent1, ...]]
        for title, sentences in zip(sample["context"]["title"], sample["context"]["sentences"]):
            text = " ".join(sentences)
            documents.append(Document(
                page_content=f"{title}\n\n{text}",
                metadata={
                    "question_id": qid,
                    "title": title,
                    "is_supporting": title in supporting_titles,
                },
            ))

        questions.append({
            "id": qid,
            "question": sample["question"],
            "ground_truth": sample["answer"],
            "type": sample["type"],    # "bridge" or "comparison"
            "level": sample["level"],  # "easy", "medium", or "hard"
        })

    print(f"\nTotal paragraphs to ingest: {len(documents)} ({n_samples} questions × 10 paragraphs)")

    print(f"\nInitialising embeddings ({EMBEDDING_MODEL})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Ingesting into ChromaDB at {HOTPOTQA_CHROMA_DIR} ...")
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=HOTPOTQA_CHROMA_DIR,
        collection_name=HOTPOTQA_COLLECTION_NAME,
    )

    BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    print(f"\nBenchmark questions saved to {BENCHMARK_PATH}")
    print(f"Ingestion complete. {len(documents)} paragraphs in collection '{HOTPOTQA_COLLECTION_NAME}'.")

    by_type = {}
    for q in questions:
        by_type.setdefault(q["type"], 0)
        by_type[q["type"]] += 1
    by_level = {}
    for q in questions:
        by_level.setdefault(q["level"], 0)
        by_level[q["level"]] += 1

    print(f"\nQuestion breakdown:")
    for k, v in sorted(by_type.items()):
        print(f"  type={k}: {v}")
    for k, v in sorted(by_level.items()):
        print(f"  level={k}: {v}")


if __name__ == "__main__":
    build_corpus()
