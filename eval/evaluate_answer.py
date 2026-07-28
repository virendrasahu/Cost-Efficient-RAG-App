import json
import numpy as np
from typing import List, Dict, Any

def evaluate_faithfulness(answer: str, context_chunks: List[str]) -> float:
    """Evaluate groundedness of answer against retrieved context chunks."""
    if not answer or answer == "I do not have sufficient information in the provided context to answer this question.":
        return 1.0  # Fallback response is faithful when info is missing
    if not context_chunks:
        return 0.0
        
    combined_context = " ".join(context_chunks).lower()
    # Simple lexical similarity check for key words as fallback evaluator
    words = [w.strip(".,!?()[]") for w in answer.lower().split() if len(w) > 3]
    if not words:
        return 1.0
    matches = sum(1 for w in words if w in combined_context)
    return round(matches / len(words), 4)

def evaluate_answer_relevance(generated_answer: str, ground_truth: str) -> float:
    """Evaluate similarity between generated answer and ground truth answer."""
    if not generated_answer or not ground_truth:
        return 0.0
    if generated_answer.strip() == ground_truth.strip():
        return 1.0
        
    gen_words = set(generated_answer.lower().split())
    gt_words = set(ground_truth.lower().split())
    intersection = gen_words.intersection(gt_words)
    union = gen_words.union(gt_words)
    return round(len(intersection) / len(union), 4) if union else 0.0

def run_answer_evaluation(eval_dataset_path: str, rag_pipeline) -> Dict[str, Any]:
    with open(eval_dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    faithfulness_scores = []
    relevance_scores = []
    details = []

    for item in dataset:
        query = item["question"]
        gold_answer = item["ground_truth_answer"]

        res = rag_pipeline.query(query_text=query, top_k=3)
        gen_answer = res["answer"]
        context_chunks = [c["text"] for c in res.get("retrieved_chunks", [])]

        faithfulness = evaluate_faithfulness(gen_answer, context_chunks)
        relevance = evaluate_answer_relevance(gen_answer, gold_answer)

        faithfulness_scores.append(faithfulness)
        relevance_scores.append(relevance)

        details.append({
            "question": query,
            "generated_answer": gen_answer,
            "ground_truth_answer": gold_answer,
            "faithfulness": faithfulness,
            "answer_relevance": relevance,
            "fallback_triggered": res.get("fallback_triggered", False)
        })

    summary = {
        "mean_faithfulness": float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0,
        "mean_answer_relevance": float(np.mean(relevance_scores)) if relevance_scores else 0.0,
        "details": details
    }
    return summary

if __name__ == "__main__":
    from src.vector_store import VectorStoreManager
    from src.rag_pipeline import RAGPipeline
    
    vs = VectorStoreManager()
    pipeline = RAGPipeline(vs)
    metrics = run_answer_evaluation("./data/eval_dataset.json", pipeline)
    print("--- Answer Quality Evaluation Metrics ---")
    print(f"Faithfulness / Groundedness: {metrics['mean_faithfulness']:.4f}")
    print(f"Answer Relevance:            {metrics['mean_answer_relevance']:.4f}")
