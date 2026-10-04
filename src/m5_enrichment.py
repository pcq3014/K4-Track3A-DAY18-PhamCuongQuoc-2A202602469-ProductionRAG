from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, re, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY

ENRICH_MODEL = "gpt-4o-mini"
ENRICH_WORKERS = 4  # số chunk enrich song song — 4 để không vượt rate limit TPM (8 luồng từng bị 429)


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


_CLIENT = None


def _chat(system: str, user: str, max_tokens: int, json_mode: bool = False) -> str:
    """Gọi gpt-4o-mini (temperature=0). Raise nếu lỗi — caller tự fallback."""
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI
        # max_retries cao: enrich song song dễ chạm rate limit TPM (429) → SDK tự backoff rồi thử lại
        _CLIENT = OpenAI(api_key=OPENAI_API_KEY, max_retries=6)
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = _CLIENT.chat.completions.create(
        model=ENRICH_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=0,
        **kwargs,
    )
    return resp.choices[0].message.content.strip()


def _document_title(text: str, source: str = "") -> str:
    """Lấy tiêu đề tài liệu (dòng '# ...' đầu tiên), fallback về tên file."""
    m = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    return m.group(1).strip() if m else source


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if OPENAI_API_KEY:
        try:
            return _chat("Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt, giữ nguyên các con số.",
                         text, max_tokens=150)
        except Exception as e:
            print(f"  ⚠️  OpenAI summarize failed: {e}")

    # Extractive fallback (không cần API): 2 câu đầu
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:2]).rstrip(".") + "." if sentences else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if OPENAI_API_KEY:
        try:
            content = _chat(f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi tiếng Việt mà đoạn văn có thể trả lời. "
                            "Trả về mỗi câu hỏi trên 1 dòng, không đánh số.",
                            text, max_tokens=200)
            questions = [q.strip().lstrip("0123456789.-) ").strip() for q in content.split("\n")]
            return [q for q in questions if q][:n_questions]
        except Exception as e:
            print(f"  ⚠️  OpenAI HyQA failed: {e}")

    # Extractive fallback: biến câu khẳng định thành câu hỏi
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if OPENAI_API_KEY:
        try:
            context = _chat("Viết 1 câu ngắn bằng tiếng Việt mô tả đoạn văn này nằm ở đâu trong tài liệu "
                            "và nói về chủ đề gì. Chỉ trả về 1 câu.",
                            f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}", max_tokens=80)
            return f"{context}\n\n{text}"
        except Exception as e:
            print(f"  ⚠️  OpenAI contextual failed: {e}")

    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if OPENAI_API_KEY:
        try:
            content = _chat('Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], '
                            '"category": "policy|hr|it|finance", "language": "vi|en"}',
                            text, max_tokens=150, json_mode=True)
            return json.loads(content)
        except Exception as e:
            print(f"  ⚠️  OpenAI metadata failed: {e}")

    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


_COMBINED_PROMPT = """Bạn làm giàu dữ liệu cho hệ thống RAG về chính sách nội bộ công ty.
Phân tích đoạn văn (trích từ tài liệu đã cho) và trả về JSON:
{
  "summary": "tóm tắt 2-3 câu, giữ nguyên con số",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu nêu tên tài liệu, phiên bản/ngày hiệu lực (nếu có) và đoạn văn nói về gì",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}
"questions" là các câu hỏi nhân viên có thể hỏi mà đoạn văn trả lời được. Tất cả bằng tiếng Việt."""


def _enrich_single_call(text: str, source: str, document_header: str = "") -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    if OPENAI_API_KEY:
        try:
            header = f"\nĐầu tài liệu:\n{document_header}" if document_header else ""
            content = _chat(_COMBINED_PROMPT, f"Tài liệu: {source}{header}\n\nĐoạn văn:\n{text}",
                            max_tokens=500, json_mode=True)
            result = json.loads(content)
            if not isinstance(result.get("metadata"), dict):
                result["metadata"] = {}
            return result
        except Exception as e:
            print(f"  ⚠️  Enrichment API failed: {e}")

    # Fallback không cần API: context = tiêu đề tài liệu, extractive summary/questions
    title = _document_title(document_header or text, source)
    return {
        "summary": summarize_chunk(text) if not OPENAI_API_KEY else "",
        "questions": [],
        "context": f"Trích từ {title}." if title else "",
        "metadata": {},
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}. metadata có thể có "document_header"
                (vài dòng đầu tài liệu) để LLM biết chunk thuộc tài liệu/phiên bản nào.
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods
    done = [0]

    def _enrich_one(chunk: dict) -> EnrichedChunk:
        text = chunk["text"]
        meta = chunk.get("metadata", {})
        source = meta.get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source, meta.get("document_header", ""))
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        done[0] += 1
        if done[0] % 10 == 0 or done[0] == len(chunks):
            print(f"  Enriched {done[0]}/{len(chunks)} chunks...", flush=True)

        return EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**meta, **auto_meta},
            method="+".join(methods),
        )

    # Enrich song song (gọi API là I/O-bound); map() giữ nguyên thứ tự chunks
    with ThreadPoolExecutor(max_workers=ENRICH_WORKERS) as pool:
        return list(pool.map(_enrich_one, chunks))


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")

    combined = _enrich_single_call(sample, "nghi_phep_nam.md")
    print(f"\nCombined (1 call): {json.dumps(combined, ensure_ascii=False, indent=2)}")
