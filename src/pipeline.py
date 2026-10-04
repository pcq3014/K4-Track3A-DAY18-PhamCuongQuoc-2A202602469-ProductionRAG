from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import os, sys, time, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K

LLM_MODEL = "gpt-4o-mini"
LATENCY_REPORT_PATH = "reports/latency_report.json"

ANSWER_SYSTEM_PROMPT = """Bạn là trợ lý tra cứu chính sách nội bộ công ty. Trả lời câu hỏi CHỈ dựa trên context được cung cấp.
Quy tắc:
- Câu đầu tiên trả lời thẳng vào câu hỏi (Có/Không, con số, người phê duyệt...), sau đó giải thích ngắn gọn 1-3 câu.
- Nếu context có nhiều phiên bản của cùng một chính sách: dùng phiên bản hiện hành (ngày hiệu lực mới nhất, không bị đánh dấu "ĐÃ THAY THẾ"), nêu rõ phiên bản, và nhắc giá trị của phiên bản cũ để đối chiếu.
- Mỗi khẳng định phải dựa trên một quy định cụ thể trong context; nêu quy định đó (kèm số liệu) làm căn cứ.
- Câu hỏi cần tính toán: nêu quy định gốc trước, quy đổi đơn vị thời gian nếu khác nhau (ngày ↔ tháng), rồi trình bày phép tính.
- Không thêm thông tin ngoài context. Nếu context không có thông tin → trả lời "Không tìm thấy thông tin trong tài liệu."
- Trả lời bằng tiếng Việt."""


def _document_header(text: str, n_lines: int = 2) -> str:
    """Tiêu đề + dòng metadata (Phiên bản | Ngày hiệu lực | ...) ở đầu tài liệu."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[:n_lines])


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)
    build_timings = {}

    # Step 1: Load & Chunk (M1) — hierarchical: index child (precision), trả về parent (context)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    parent_store: dict[str, str] = {}  # "source::parent_id" → parent text
    for doc in docs:
        source = doc["metadata"]["source"]
        header = _document_header(doc["text"])
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for parent in parents:
            parent_store[f"{source}::{parent.metadata['parent_id']}"] = parent.text
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {
                **child.metadata,
                "parent_id": child.parent_id,
                "parent_key": f"{source}::{child.parent_id}",
                "document_header": header,
            }})
    build_timings["chunking_s"] = time.time() - t0
    print(f"  ✓ {len(all_chunks)} child chunks / {len(parent_store)} parents from {len(docs)} documents "
          f"({build_timings['chunking_s']:.1f}s)", flush=True)

    # Step 2: Enrichment (M5) — combined mode, 1 API call/chunk
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        index_chunks = []
        for e in enriched:
            header = e.auto_metadata.get("document_header", "")
            # Index = header tài liệu + context line + chunk + HyQA questions (bridge vocabulary gap)
            parts = [header, e.enriched_text]
            if e.hypothesis_questions:
                parts.append("Câu hỏi liên quan: " + " | ".join(e.hypothesis_questions))
            index_chunks.append({"text": "\n".join(p for p in parts if p), "metadata": e.auto_metadata})
        all_chunks = index_chunks
        build_timings["enrichment_s"] = time.time() - t0
        print(f"  ✓ Enriched {len(enriched)} chunks ({build_timings['enrichment_s']:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    search.parent_store = parent_store
    build_timings["indexing_s"] = time.time() - t0
    print(f"  ✓ Indexed ({build_timings['indexing_s']:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    reranker._load_model()
    build_timings["reranker_load_s"] = time.time() - t0
    print(f"  ✓ Reranker ready ({build_timings['reranker_load_s']:.1f}s)", flush=True)

    search.build_timings = build_timings
    return search, reranker


def _children_to_parents(results, parent_store: dict[str, str]) -> list[dict]:
    """Child hits → parent docs (dedupe, giữ thứ tự hybrid rank đầu tiên của mỗi parent)."""
    parents, seen = [], set()
    for r in results:
        key = r.metadata.get("parent_key")
        if key in seen:
            continue
        if key and key in parent_store:
            seen.add(key)
            parents.append({"text": parent_store[key], "score": r.score,
                            "metadata": {"source": r.metadata.get("source", ""), "parent_key": key}})
        elif not key:
            parents.append({"text": r.text, "score": r.score, "metadata": r.metadata})
    return parents


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker,
              timings: dict | None = None) -> tuple[str, list[str]]:
    """Run single query through pipeline. Ghi latency từng bước vào `timings` (nếu truyền vào)."""
    timings = timings if timings is not None else {}

    t0 = time.perf_counter()
    results = search.search(query)
    timings["hybrid_search_ms"] = (time.perf_counter() - t0) * 1000

    parent_store = getattr(search, "parent_store", {})
    docs = _children_to_parents(results, parent_store) if parent_store else \
        [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]

    t0 = time.perf_counter()
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
    contexts = [r.text for r in reranked] if reranked else [d["text"] for d in docs[:RERANK_TOP_K]]
    sources = [r.metadata.get("source", "") for r in reranked]

    t0 = time.perf_counter()
    from config import OPENAI_API_KEY
    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI
            client = OpenAI()
            context_str = "\n\n".join(
                f"[Tài liệu {i + 1}{': ' + sources[i] if i < len(sources) and sources[i] else ''}]\n{c}"
                for i, c in enumerate(contexts))
            resp = client.chat.completions.create(model=LLM_MODEL, temperature=0, messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ])
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    timings["generation_ms"] = (time.perf_counter() - t0) * 1000
    return answer, contexts


def _save_latency_report(build_timings: dict, query_timings: list[dict], eval_s: float):
    """Latency breakdown: thời gian build từng bước + trung bình/max mỗi bước khi query."""
    steps = ["hybrid_search_ms", "rerank_ms", "generation_ms"]
    per_step = {}
    for step in steps:
        values = [t[step] for t in query_timings if step in t]
        if values:
            per_step[step] = {"avg": sum(values) / len(values), "min": min(values), "max": max(values)}
    totals = [sum(t.get(s, 0) for s in steps) for t in query_timings]
    report = {
        "build": build_timings,
        "query": per_step,
        "query_total_ms": {"avg": sum(totals) / len(totals), "max": max(totals)} if totals else {},
        "ragas_eval_s": eval_s,
        "num_queries": len(query_timings),
    }
    os.makedirs(os.path.dirname(LATENCY_REPORT_PATH), exist_ok=True)
    with open(LATENCY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("LATENCY BREAKDOWN")
    print("=" * 60)
    for name, secs in build_timings.items():
        print(f"  [build] {name:<20} {secs:>9.1f} s")
    print(f"  {'[query] step':<28} {'avg ms':>9} {'max ms':>9}")
    for step, s in per_step.items():
        print(f"  [query] {step:<20} {s['avg']:>9.1f} {s['max']:>9.1f}")
    if totals:
        print(f"  [query] {'total':<20} {report['query_total_ms']['avg']:>9.1f} {report['query_total_ms']['max']:>9.1f}")
    print(f"Latency report saved to {LATENCY_REPORT_PATH}")


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []
    query_timings = []

    for i, item in enumerate(test_set):
        timings = {}
        answer, contexts = run_query(item["question"], search, reranker, timings)
        query_timings.append(timings)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    eval_s = time.time() - t0
    print(f"  ✓ RAGAS done ({eval_s:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    _save_latency_report(getattr(search, "build_timings", {}), query_timings, eval_s)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
