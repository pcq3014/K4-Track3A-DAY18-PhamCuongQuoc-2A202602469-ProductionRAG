from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
JUDGE_MODEL = "gpt-4o-mini"
JUDGE_EMBEDDING_MODEL = "text-embedding-3-small"
PROMPT_LANGUAGE = "vietnamese"
PROMPT_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ragas_prompts")


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _vietnamese_answer_relevancy():
    """answer_relevancy với prompt sinh câu hỏi tiếng Việt (load từ cache trong repo).

    Metric gốc sinh câu hỏi từ answer bằng prompt tiếng Anh → câu hỏi tiếng Anh, rồi so cosine
    với câu hỏi gốc tiếng Việt → điểm bị kéo thấp do lệch ngôn ngữ, không phải do answer kém
    (đo trên cùng 20 answers: EN prompt 0.645 vs VI prompt 0.814).

    Prompt được dịch 1 lần bằng scripts/build_ragas_prompts.py rồi lưu ở ragas_prompts/ — không adapt
    lúc chạy (LLM dịch không ổn định, có lần ra JSON lỗi) → baseline & production luôn cùng 1 prompt.
    """
    from ragas.llms.prompt import Prompt
    from ragas.metrics import answer_relevancy
    from ragas.metrics._answer_relevance import AnswerRelevancy

    metric = AnswerRelevancy()
    name = metric.question_generation.name
    if not os.path.exists(os.path.join(PROMPT_CACHE_DIR, PROMPT_LANGUAGE, f"{name}.json")):
        print(f"  ⚠️  Thiếu prompt tiếng Việt ({PROMPT_CACHE_DIR}) — chạy scripts/build_ragas_prompts.py. "
              "Dùng prompt gốc (EN).")
        return answer_relevancy
    metric.question_generation = Prompt._load(PROMPT_LANGUAGE, name, PROMPT_CACHE_DIR)
    return metric


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    zeros = {m: 0.0 for m in METRICS}
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, context_precision, context_recall
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        # Chỉ định judge model rõ ràng — không phụ thuộc default của RAGAS (có thể là model đã deprecated)
        llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0))
        embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=JUDGE_EMBEDDING_MODEL))
        result = evaluate(dataset, metrics=[faithfulness, _vietnamese_answer_relevancy(),
                                            context_precision, context_recall],
                          llm=llm, embeddings=embeddings, raise_exceptions=False)
        df = result.to_pandas()

        def _score(row, metric):
            value = row.get(metric, 0.0)
            return 0.0 if value is None or math.isnan(float(value)) else float(value)

        per_question = [EvalResult(
            question=row["question"], answer=row["answer"], contexts=list(row["contexts"]),
            ground_truth=row["ground_truth"],
            faithfulness=_score(row, "faithfulness"),
            answer_relevancy=_score(row, "answer_relevancy"),
            context_precision=_score(row, "context_precision"),
            context_recall=_score(row, "context_recall"),
        ) for _, row in df.iterrows()]

        # Aggregate = mean bỏ qua NaN (câu bị lỗi judge không kéo điểm về 0)
        aggregate = {m: float(df[m].mean(skipna=True)) if m in df else 0.0 for m in METRICS}
        aggregate = {m: 0.0 if math.isnan(v) else v for m, v in aggregate.items()}
        return {**aggregate, "per_question": per_question}
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return {**zeros, "per_question": []}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    # Diagnostic Tree: metric thấp nhất → bước hỏng trong pipeline → cách sửa
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating — answer chứa thông tin không có trong context",
                         "Tighten prompt (chỉ dùng context, trích dẫn), temperature=0"),
        "context_recall": ("Missing relevant chunks — retriever bỏ sót thông tin cần cho ground truth",
                           "Improve chunking (parent-child) hoặc thêm BM25/HyQA, tăng top_k"),
        "context_precision": ("Too many irrelevant chunks — context nhiễu hoặc chunk đúng bị xếp thấp",
                              "Add reranking hoặc metadata filter (version/ngày hiệu lực)"),
        "answer_relevancy": ("Answer doesn't match question — trả lời lan man/lệch trọng tâm",
                             "Improve prompt template: trả lời trực tiếp, ngắn gọn đúng câu hỏi"),
    }

    analyzed = []
    for r in eval_results:
        scores = {m: getattr(r, m) for m in METRICS}
        worst_metric = min(scores, key=scores.get)
        diagnosis, fix = diagnostic_tree[worst_metric]
        analyzed.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "scores": scores,
            "avg_score": sum(scores.values()) / len(scores),
            "worst_metric": worst_metric,
            "score": scores[worst_metric],
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })

    analyzed.sort(key=lambda x: x["avg_score"])
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
        "per_question": [asdict(r) if isinstance(r, EvalResult) else r
                         for r in results.get("per_question", [])],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
