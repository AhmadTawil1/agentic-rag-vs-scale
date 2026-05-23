"""
Benchmark loader for the HotpotQA evaluation.

Run `python ingestion/ingest_hotpotqa.py` first to generate benchmark_questions.json.
"""

import json
from pathlib import Path

BENCHMARK_PATH = Path(__file__).parent / "benchmark_questions.json"


def load_benchmark() -> list[dict]:
    """
    Load the sampled HotpotQA questions.

    Returns a list of dicts with keys:
        id, question, ground_truth, type ("bridge" | "comparison"), level
    """
    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(
            f"{BENCHMARK_PATH} not found. "
            "Run `python ingestion/ingest_hotpotqa.py` first."
        )
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    questions = load_benchmark()
    by_type = {}
    by_level = {}
    for q in questions:
        by_type.setdefault(q["type"], 0)
        by_type[q["type"]] += 1
        by_level.setdefault(q["level"], 0)
        by_level[q["level"]] += 1

    print(f"Benchmark: {len(questions)} questions")
    for k, v in sorted(by_type.items()):
        print(f"  type={k}: {v}")
    for k, v in sorted(by_level.items()):
        print(f"  level={k}: {v}")
