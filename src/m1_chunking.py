from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    from numpy import dot
    from numpy.linalg import norm

    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s and s.strip()]
    if not sentences:
        return []

    embeddings = _get_semantic_model().encode(sentences)

    def cosine_sim(a, b):
        return float(dot(a, b) / (norm(a) * norm(b) + 1e-9))

    # Câu mới gần nghĩa với câu trước → gộp; tụt dưới threshold → mở chunk mới
    groups = [[sentences[0]]]
    for i in range(1, len(sentences)):
        if cosine_sim(embeddings[i - 1], embeddings[i]) < threshold:
            groups.append([sentences[i]])
        else:
            groups[-1].append(sentences[i])

    return [Chunk(text=" ".join(g), metadata={**metadata, "strategy": "semantic", "chunk_index": i})
            for i, g in enumerate(groups)]


_SEMANTIC_MODEL = None


def _get_semantic_model():
    """Load embedding model 1 lần duy nhất (tránh reload mỗi lần gọi chunk_semantic)."""
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SEMANTIC_MODEL


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Gộp paragraphs thành parents ≤ parent_size (paragraph quá dài thì đứng riêng 1 parent)
    parent_texts, current = [], ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > parent_size:
            parent_texts.append(current)
            current = ""
        current = f"{current}\n\n{para}" if current else para
    if current:
        parent_texts.append(current)

    parents, children = [], []
    for ptext in parent_texts:
        pid = f"parent_{len(parents)}"
        parents.append(Chunk(text=ptext, metadata={**metadata, "chunk_type": "parent", "parent_id": pid}))
        for ctext in _split_to_size(ptext, child_size):
            children.append(Chunk(text=ctext,
                                  metadata={**metadata, "chunk_type": "child", "child_index": len(children)},
                                  parent_id=pid))
    return (parents, children)


def _split_to_size(text: str, size: int) -> list[str]:
    """Chia text thành các đoạn ≤ size: ưu tiên ranh giới dòng/câu, chỉ cắt theo từ khi 1 câu quá dài."""
    pieces = []
    for unit in re.split(r'\n+|(?<=[.!?])\s+', text):
        unit = unit.strip()
        if not unit:
            continue
        if len(unit) <= size:
            pieces.append(unit)
            continue
        words_buf = ""
        for word in unit.split():
            if words_buf and len(words_buf) + len(word) + 1 > size:
                pieces.append(words_buf)
                words_buf = ""
            words_buf = f"{words_buf} {word}" if words_buf else word
        if words_buf:
            pieces.append(words_buf)

    # Gộp lại các mảnh nhỏ liền nhau sao cho mỗi child vẫn ≤ size
    out, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) + 1 > size:
            out.append(current)
            current = ""
        current = f"{current}\n{piece}" if current else piece
    if current:
        out.append(current)
    return out


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)
    header_re = re.compile(r'^(#{1,3})\s+(.+)$')

    chunks: list[Chunk] = []
    header_path: list[tuple[int, str]] = []  # breadcrumb [(level, title)], vd: h1 > h2
    current_header, current_content = "", ""

    def flush():
        content = current_content.strip()
        if not content:
            return  # header không có nội dung (h1 nằm ngay trên h2) → chỉ giữ trong breadcrumb
        body = f"{current_header}\n{content}" if current_header else content
        chunks.append(Chunk(text=body, metadata={
            **metadata,
            "section": current_header.lstrip("#").strip(),
            "section_path": " > ".join(title for _, title in header_path),
            "strategy": "structure",
            "chunk_index": len(chunks),
        }))

    for part in sections:
        m = header_re.match(part)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            header_path = [(lv, t) for lv, t in header_path if lv < level] + [(level, title)]
            current_header, current_content = part.strip(), ""
        else:
            current_content += part
    flush()
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
