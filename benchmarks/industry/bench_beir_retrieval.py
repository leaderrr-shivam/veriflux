"""BEIR & MS-MARCO Information Retrieval Industry Benchmark.

Evaluates Veriflux BM25+ Ranking, TF-IDF Vectorization, and Reciprocal Rank Fusion (RRF)
against 50 technical multi-domain queries across 250 corpus documents.
Computes standard academic IR metrics: MRR@10, Hit@1, Hit@3, Hit@5, and NDCG@10.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from veriflux.similarity.bm25 import BM25
from veriflux.similarity.fusion import reciprocal_rank_fusion
from veriflux.similarity.tfidf import TfidfVectorizer

DATASET_PATH = Path(__file__).resolve().parent.parent / "datasets" / "beir_msmarco_subset.json"


def compute_dcg(relevances: List[int], k: int = 10) -> float:
    dcg = 0.0
    for idx, rel in enumerate(relevances[:k]):
        dcg += (2**rel - 1) / math.log2(idx + 2)
    return dcg


def run_beir_benchmark() -> Dict[str, Any]:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    corpus = data["corpus"]
    queries = data["queries"]

    doc_ids = [d["id"] for d in corpus]
    doc_texts = [d["text"] for d in corpus]

    bm25 = BM25(k1=1.5, b=0.75, delta=0.5)
    bm25.fit(doc_texts)

    tfidf = TfidfVectorizer()
    tfidf_vectors = tfidf.fit_transform(doc_texts)

    mrr_bm, hit1_bm, hit3_bm, hit5_bm, hit10_bm, ndcg_bm = 0.0, 0, 0, 0, 0, 0.0
    mrr_hyb, hit1_hyb, hit3_hyb, hit5_hyb, hit10_hyb, ndcg_hyb = 0.0, 0, 0, 0, 0, 0.0

    start_time = time.perf_counter()

    for q in queries:
        q_text = q["query"]
        target_id = q["relevant_doc_id"]
        target_idx = doc_ids.index(target_id)

        # 1. BM25+ Ranking
        bm_rankings = bm25.rank(q_text, top_k=20)
        bm_ranked_ids = [doc_ids[idx] for idx, _ in bm_rankings[:10]]

        if target_id in bm_ranked_ids:
            rank = bm_ranked_ids.index(target_id) + 1
            mrr_bm += 1.0 / rank
            if rank == 1: hit1_bm += 1
            if rank <= 3: hit3_bm += 1
            if rank <= 5: hit5_bm += 1
            hit10_bm += 1

            rels = [1 if d_id == target_id else 0 for d_id in bm_ranked_ids]
            dcg = compute_dcg(rels, k=10)
            idcg = compute_dcg([1], k=10)
            ndcg_bm += dcg / idcg if idcg > 0 else 0.0

        # 2. Hybrid RRF (BM25+ and TF-IDF Cosine)
        qv = tfidf.transform([q_text])[0]
        tfidf_scores = [(idx, qv.cosine_with(vd)) for idx, vd in enumerate(tfidf_vectors)]
        tfidf_scores.sort(key=lambda x: x[1], reverse=True)
        tfidf_rankings = tfidf_scores[:20]

        fused = reciprocal_rank_fusion([bm_rankings, tfidf_rankings], k=30)
        hyb_ranked_ids = [doc_ids[idx] for idx, _ in fused[:10]]

        if target_id in hyb_ranked_ids:
            rank = hyb_ranked_ids.index(target_id) + 1
            mrr_hyb += 1.0 / rank
            if rank == 1: hit1_hyb += 1
            if rank <= 3: hit3_hyb += 1
            if rank <= 5: hit5_hyb += 1
            hit10_hyb += 1

            rels = [1 if d_id == target_id else 0 for d_id in hyb_ranked_ids]
            dcg = compute_dcg(rels, k=10)
            idcg = compute_dcg([1], k=10)
            ndcg_hyb += dcg / idcg if idcg > 0 else 0.0

    total_time = time.perf_counter() - start_time
    total_q = len(queries)
    latency_ms = (total_time / total_q) * 1000.0
    qps = total_q / max(0.0001, total_time)

    report = {
        "benchmark": "BEIR / MS-MARCO Information Retrieval Evaluation",
        "total_queries": total_q,
        "corpus_documents": len(corpus),
        "mean_latency_ms": round(latency_ms, 3),
        "throughput_qps": round(qps, 1),
        "bm25_plus": {
            "mrr_at_10": round(mrr_bm / total_q, 4),
            "ndcg_at_10": round(ndcg_bm / total_q, 4),
            "hit_at_1": round((hit1_bm / total_q) * 100.0, 1),
            "hit_at_3": round((hit3_bm / total_q) * 100.0, 1),
            "hit_at_5": round((hit5_bm / total_q) * 100.0, 1),
            "hit_at_10": round((hit10_bm / total_q) * 100.0, 1),
        },
        "hybrid_rrf": {
            "mrr_at_10": round(mrr_hyb / total_q, 4),
            "ndcg_at_10": round(ndcg_hyb / total_q, 4),
            "hit_at_1": round((hit1_hyb / total_q) * 100.0, 1),
            "hit_at_3": round((hit3_hyb / total_q) * 100.0, 1),
            "hit_at_5": round((hit5_hyb / total_q) * 100.0, 1),
            "hit_at_10": round((hit10_hyb / total_q) * 100.0, 1),
        },
    }

    return report


if __name__ == "__main__":
    rep = run_beir_benchmark()
    print("=" * 65)
    print(f"  {rep['benchmark']}")
    print("=" * 65)
    print(f" Total Queries Evaluated   : {rep['total_queries']}")
    print(f" Corpus Documents          : {rep['corpus_documents']}")
    print(f" Mean Latency per Query    : {rep['mean_latency_ms']} ms")
    print(f" Retrieval Throughput      : {rep['throughput_qps']:,} QPS")
    print("-" * 65)
    print(" [1] Pure BM25+ Ranking Metrics:")
    print(f"     MRR@10 : {rep['bm25_plus']['mrr_at_10']}")
    print(f"     NDCG@10: {rep['bm25_plus']['ndcg_at_10']}")
    print(f"     Hit@1  : {rep['bm25_plus']['hit_at_1']}%")
    print(f"     Hit@5  : {rep['bm25_plus']['hit_at_5']}%")
    print(f"     Hit@10 : {rep['bm25_plus']['hit_at_10']}%")
    print("-" * 65)
    print(" [2] Hybrid Fusion (BM25+ + TF-IDF RRF) Metrics:")
    print(f"     MRR@10 : {rep['hybrid_rrf']['mrr_at_10']}")
    print(f"     NDCG@10: {rep['hybrid_rrf']['ndcg_at_10']}")
    print(f"     Hit@1  : {rep['hybrid_rrf']['hit_at_1']}%")
    print(f"     Hit@5  : {rep['hybrid_rrf']['hit_at_5']}%")
    print(f"     Hit@10 : {rep['hybrid_rrf']['hit_at_10']}%")
    print("=" * 65)
