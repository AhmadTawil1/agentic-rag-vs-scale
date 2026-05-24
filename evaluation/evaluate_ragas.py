"""
RAGAS Evaluation: Score all four systems using RAGAS metrics.
Evaluator: GPT-5  |  Compatible with RAGAS 0.4.x
"""

import sys
import json
import asyncio
from pathlib import Path

# Windows fix: RAGAS uses asyncio internally and the default ProactorEventLoop
# causes coroutine warnings and hangs on Windows.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))

from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import EVALUATOR_MODEL


def load_results(results_path):
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_ragas_dataset(results):
    """Convert results list to RAGAS 0.4.x EvaluationDataset, skipping ERROR entries."""
    from ragas import SingleTurnSample

    samples = []
    for r in results:
        if r["answer"].startswith("ERROR"):
            continue
        samples.append(
            SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=r["contexts"],
                reference=r["ground_truth"],
            )
        )
    return EvaluationDataset(samples=samples)


def evaluate_system(results_path, system_name):
    """Evaluate a single system using RAGAS metrics. Returns dict of metric -> mean score."""
    print(f"\n{'='*80}")
    print(f"EVALUATING {system_name}")
    print(f"{'='*80}")

    results = load_results(results_path)
    print(f"Loaded {len(results)} results")

    dataset = prepare_ragas_dataset(results)
    print(f"Prepared {len(dataset.samples)} valid samples for evaluation")

    llm = LangchainLLMWrapper(ChatOpenAI(model=EVALUATOR_MODEL, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    print("\nRunning RAGAS evaluation...")
    print("Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall")

    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
        llm=llm,
        embeddings=embeddings,
    )

    df = result.to_pandas()
    scores = {
        "faithfulness": df["faithfulness"].mean(),
        "answer_relevancy": df["answer_relevancy"].mean(),
        "context_precision": df["context_precision"].mean(),
        "context_recall": df["context_recall"].mean(),
    }

    print(f"\n{'-'*80}")
    print(f"RESULTS FOR {system_name}")
    print(f"{'-'*80}")
    for metric, score in scores.items():
        print(f"{metric:20s}: {score:.4f}")

    return scores


def compare_systems(results_a_path, results_b_path, results_c_path=None, results_d_path=None):
    """
    Evaluate and compare up to four systems.

    System A - Goliath  (Naive RAG + GPT-5.4-mini)
    System B - David    (Agentic RAG + Llama)
    System C - Hermes   (Naive RAG + Llama)          [optional]
    System D - Titan    (Agentic RAG + GPT-5.4-mini) [optional]
    """
    print("=" * 80)
    print("RAG SYSTEMS EVALUATION WITH RAGAS")
    print(f"Evaluator: {EVALUATOR_MODEL}")
    print("=" * 80)

    systems: dict[str, dict] = {}
    systems["System A (Goliath - Naive GPT)"] = evaluate_system(
        results_a_path, "SYSTEM A (Goliath - Naive GPT)"
    )
    systems["System B (David - Agentic Llama)"] = evaluate_system(
        results_b_path, "SYSTEM B (David - Agentic Llama)"
    )
    if results_c_path:
        systems["System C (Hermes - Naive Llama)"] = evaluate_system(
            results_c_path, "SYSTEM C (Hermes - Naive Llama)"
        )
    if results_d_path:
        systems["System D (Titan - Agentic GPT)"] = evaluate_system(
            results_d_path, "SYSTEM D (Titan - Agentic GPT)"
        )

    # --- Comparison table ---
    metrics = list(next(iter(systems.values())).keys())
    names = list(systems.keys())
    short = {
        "System A (Goliath - Naive GPT)":   "A:NaiveGPT",
        "System B (David - Agentic Llama)":  "B:AgentLlama",
        "System C (Hermes - Naive Llama)":   "C:NaiveLlama",
        "System D (Titan - Agentic GPT)":    "D:AgentGPT",
    }
    col_w = 14

    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    header = f"{'Metric':<22s}" + "".join(f"{short.get(n,n):>{col_w}s}" for n in names) + f"{'Winner':>14s}"
    print(header)
    print("-" * len(header))

    winners: dict[str, int] = {n: 0 for n in names}
    for metric in metrics:
        scores = {n: systems[n][metric] for n in names}
        best = max(scores, key=scores.__getitem__)
        winners[best] += 1
        row = (
            f"{metric:<22s}"
            + "".join(f"{scores[n]:>{col_w}.4f}" for n in names)
            + f"{short.get(best, best):>14s}"
        )
        print(row)

    print(f"\n{'='*80}")
    for name, wins in winners.items():
        print(f"{name}: {wins}/{len(metrics)} metrics won")

    overall_winner = max(winners, key=winners.__getitem__)
    print(f"\nWINNER: {overall_winner}")
    print("=" * 80)

    # --- Save CSV ---
    comparison_df = pd.DataFrame(
        {"Metric": metrics, **{n: [systems[n][m] for m in metrics] for n in names}}
    )
    comparison_path = Path(results_a_path).parent / "comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nComparison saved to: {comparison_path}")

    return systems


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG systems with RAGAS")
    parser.add_argument("--system-a", required=True, help="Path to System A results JSON")
    parser.add_argument("--system-b", required=True, help="Path to System B results JSON")
    parser.add_argument("--system-c", default=None, help="Path to System C results JSON (optional)")
    parser.add_argument("--system-d", default=None, help="Path to System D results JSON (optional)")

    args = parser.parse_args()
    compare_systems(args.system_a, args.system_b, args.system_c, args.system_d)
