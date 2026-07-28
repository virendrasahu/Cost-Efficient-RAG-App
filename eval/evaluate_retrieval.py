import json
import numpy as np
from typing import List, Dict, Any

def compute_hit_rate(retrieved_sources: List[str], gold_source: str, k: int) -> float:
    """Compute Hit Rate@k for a single query."""
    if not gold_source:
        return 1.0
    top_k_sources = set(retrieved_sources[:k])
    return 1.0 if gold_source in top_k_sources else 0.0

def compute_recall_at_k(retrieved_sources: List[str], gold_source: str, k: int) -> float:
    """Compute Recall@k for a single query."""
    if not gold_source:
        return 1.0
    top_k_sources = retrieved_sources[:k]
    hits = sum(1 for src in top_k_sources if src == gold_source)
    return min(1.0, hits / 1.0)

def compute_mrr(retrieved_sources: List[str], gold_source: str) -> float:
    """Compute Mean Reciprocal Rank (MRR) for a single query."""
    if not gold_source:
        return 1.0
    for idx, source in enumerate(retrieved_sources):
        if source == gold_source:
            return 1.0 / (idx + 1)
    return 0.0

def compute_ndcg_at_k(retrieved_sources: List[str], gold_source: str, k: int) -> float:
    """Compute nDCG@k for binary relevance (first match of target source)."""
    if not gold_source:
        return 1.0
    top_k = retrieved_sources[:k]
    for idx, source in enumerate(top_k):
        if source == gold_source:
            return float(1.0 / np.log2(idx + 2))
    return 0.0

def compute_context_precision(retrieved_sources: List[str], gold_source: str, k: int) -> float:
    """Compute Context Precision@k based on cumulative precision at relevant ranks."""
    if not gold_source:
        return 1.0
    top_k = retrieved_sources[:k]
    relevant_count = 0
    precision_sum = 0.0

    for idx, source in enumerate(top_k):
        if source == gold_source:
            relevant_count += 1
            precision_at_k = relevant_count / (idx + 1)
            precision_sum += precision_at_k

    return precision_sum / relevant_count if relevant_count > 0 else 0.0

def run_retrieval_evaluation(eval_dataset_path: str, rag_pipeline, k: int = 3) -> Dict[str, Any]:
    with open(eval_dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    hit_rates = []
    recalls = []
    mrrs = []
    ndcgs = []
    precisions = []

    results_detail = []

    for item in dataset:
        query = item["question"]
        gold_source = item.get("gold_source")

        result = rag_pipeline.query(query_text=query, top_k=k)
        retrieved_sources = [c["source"] for c in result.get("retrieved_chunks", [])]

        if gold_source:
            hit = compute_hit_rate(retrieved_sources, gold_source, k)
            r_k = compute_recall_at_k(retrieved_sources, gold_source, k)
            mrr = compute_mrr(retrieved_sources, gold_source)
            ndcg = compute_ndcg_at_k(retrieved_sources, gold_source, k)
            cp = compute_context_precision(retrieved_sources, gold_source, k)

            hit_rates.append(hit)
            recalls.append(r_k)
            mrrs.append(mrr)
            ndcgs.append(ndcg)
            precisions.append(cp)

            results_detail.append({
                "question": query,
                "gold_source": gold_source,
                "retrieved_sources": retrieved_sources,
                "hit_rate": hit,
                "recall_at_k": r_k,
                "mrr": mrr,
                "ndcg_at_k": ndcg,
                "context_precision": cp
            })

    summary = {
        "mean_hit_rate": float(np.mean(hit_rates)) if hit_rates else 0.0,
        "mean_recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "mean_mrr": float(np.mean(mrrs)) if mrrs else 0.0,
        "mean_ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "mean_context_precision": float(np.mean(precisions)) if precisions else 0.0,
        "details": results_detail
    }
    return summary

if __name__ == "__main__":
    from src.vector_store import VectorStoreManager
    from src.rag_pipeline import RAGPipeline
    from src.ingestion import process_and_deduplicate
    
    vs = VectorStoreManager()
    existing = vs.get_existing_ids()
    records = process_and_deduplicate("./data/raw_documents/guide.md", existing)
    vs.add_chunks(records)

    pipeline = RAGPipeline(vs)
    metrics = run_retrieval_evaluation("./data/eval_dataset.json", pipeline, k=3)
    
    # Save results to results/eval_results.json
    with open("results/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("--- Retrieval Evaluation Metrics ---")
    print(f"Hit Rate@3:          {metrics['mean_hit_rate']:.4f}")
    print(f"Recall@3:            {metrics['mean_recall_at_k']:.4f}")
    print(f"MRR:                 {metrics['mean_mrr']:.4f}")
    print(f"nDCG@3:              {metrics['mean_ndcg_at_k']:.4f}")
    print(f"Context Precision:   {metrics['mean_context_precision']:.4f}")
