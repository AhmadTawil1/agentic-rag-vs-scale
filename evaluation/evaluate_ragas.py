"""
RAGAS Evaluation: Score both systems using RAGAS metrics
"""

import sys
import json
import os
from pathlib import Path
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))

from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from langchain_openai import ChatOpenAI


def load_results(results_path):
    """Load results from JSON file."""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_ragas_dataset(results):
    """
    Convert results to RAGAS dataset format.
    
    RAGAS expects:
    - question: str
    - answer: str
    - contexts: List[str]
    - ground_truth: str
    """
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    for result in results:
        # Skip errors
        if result["answer"].startswith("ERROR"):
            continue
        
        data["question"].append(result["question"])
        data["answer"].append(result["answer"])
        data["contexts"].append(result["contexts"])
        data["ground_truth"].append(result["ground_truth"])
    
    return Dataset.from_dict(data)


def evaluate_system(results_path, system_name):
    """
    Evaluate a system using RAGAS metrics.
    
    Args:
        results_path: Path to results JSON file
        system_name: Name of the system (for display)
        
    Returns:
        Dictionary of metric scores
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING {system_name}")
    print(f"{'='*80}")
    
    # Load results
    results = load_results(results_path)
    print(f"Loaded {len(results)} results")
    
    # Prepare dataset
    dataset = prepare_ragas_dataset(results)
    print(f"Prepared {len(dataset)} valid samples for evaluation")
    
    # Initialize LLM for RAGAS (uses GPT-4o-mini for evaluation)
    llm = ChatOpenAI(model="gpt-5.4", temperature=0)
    
    # Run evaluation
    print("\nRunning RAGAS evaluation...")
    print("Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall")
    
    evaluation_result = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall()
        ],
        llm=llm
    )
    
    # Extract scores (RAGAS returns per-sample scores, we need to aggregate)
    import numpy as np
    
    scores = {
        "faithfulness": np.mean(evaluation_result["faithfulness"]),
        "answer_relevancy": np.mean(evaluation_result["answer_relevancy"]),
        "context_precision": np.mean(evaluation_result["context_precision"]),
        "context_recall": np.mean(evaluation_result["context_recall"])
    }
    
    # Display results
    print(f"\n{'-'*80}")
    print(f"RESULTS FOR {system_name}")
    print(f"{'-'*80}")
    for metric, score in scores.items():
        print(f"{metric:20s}: {score:.4f}")
    
    return scores


def compare_systems(results_a_path, results_b_path):
    """
    Evaluate and compare both systems.
    
    Args:
        results_a_path: Path to System A results
        results_b_path: Path to System B results
    """
    print("=" * 80)
    print("RAG SYSTEMS EVALUATION WITH RAGAS")
    print("=" * 80)
    
    # Evaluate System A
    scores_a = evaluate_system(results_a_path, "SYSTEM A (Goliath - Baseline)")
    
    # Evaluate System B
    scores_b = evaluate_system(results_b_path, "SYSTEM B (David - Agentic)")
    
    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    print(f"{'Metric':<20s} {'System A':>12s} {'System B':>12s} {'Winner':>12s}")
    print(f"{'-'*80}")
    
    for metric in scores_a.keys():
        a_score = scores_a[metric]
        b_score = scores_b[metric]
        winner = "System A" if a_score > b_score else "System B" if b_score > a_score else "Tie"
        
        print(f"{metric:<20s} {a_score:>12.4f} {b_score:>12.4f} {winner:>12s}")
    
    # Overall winner
    a_wins = sum(1 for metric in scores_a.keys() if scores_a[metric] > scores_b[metric])
    b_wins = sum(1 for metric in scores_a.keys() if scores_b[metric] > scores_a[metric])
    
    print(f"\n{'='*80}")
    print(f"System A wins: {a_wins}/4 metrics")
    print(f"System B wins: {b_wins}/4 metrics")
    
    if a_wins > b_wins:
        print("\n🏆 WINNER: System A (Goliath - Baseline)")
    elif b_wins > a_wins:
        print("\n🏆 WINNER: System B (David - Agentic)")
    else:
        print("\n🤝 RESULT: Tie")
    
    print("=" * 80)
    
    # Save comparison
    comparison_df = pd.DataFrame({
        "Metric": list(scores_a.keys()),
        "System_A": list(scores_a.values()),
        "System_B": list(scores_b.values())
    })
    
    comparison_path = Path(results_a_path).parent / "comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nComparison saved to: {comparison_path}")
    
    return scores_a, scores_b


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG systems with RAGAS")
    parser.add_argument("--system-a", required=True, help="Path to System A results JSON")
    parser.add_argument("--system-b", required=True, help="Path to System B results JSON")
    
    args = parser.parse_args()
    
    compare_systems(args.system_a, args.system_b)
