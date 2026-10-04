# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Phạm Cường Quốc
**Khóa:** K4 - Track 3A
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Trên toàn corpus (26 tài liệu), threshold 0.85 tạo **208 chunks, avg 99 ký tự** (min 6) so với basic **51 chunks, avg 410**. Model `all-MiniLM-L6-v2` là model tiếng Anh nên cosine giữa các câu tiếng Việt thấp và dao động → cắt quá vụn (có chunk 6 ký tự). Kết luận: semantic chunking phụ thuộc mạnh vào embedding model; với tiếng Việt cần model đa ngôn ngữ (bge-m3) hoặc hạ threshold. |
| Hierarchical (parent-child) | M1 | `chunk_hierarchical()` | Đây là strategy dùng trong pipeline: **104 children (≤256 ký tự) / 26 parents**. Tài liệu ngắn (~1KB) nên mỗi parent = cả 1 chính sách. Search trên child (khớp chính xác), trả **parent** cho LLM → context đủ ý cho câu multi-hop/tính toán. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | **106 chunks, avg 196 ký tự**, mỗi chunk = 1 mục `##` kèm header và `section_path` (breadcrumb h1 > h2). Bảng (lương, thẩm quyền mua sắm) không bị cắt giữa chừng — max 788 ký tự là chunk chứa nguyên bảng. |
| BM25 + Dense fusion | M2 | `segment_vietnamese()`, `reciprocal_rank_fusion()` | underthesea nối từ ghép bằng `_` ("nghỉ_phép") → phải `replace("_", " ")` + lowercase + bỏ dấu câu thì query "nghỉ phép" mới khớp. RRF (k=60) chỉ dùng **thứ hạng**, không cần chuẩn hoá điểm BM25 (0–20) với cosine (0–1) — BM25 bắt từ khoá/con số chính xác ("MFA", "55 triệu"), dense bắt diễn đạt khác nghĩa. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | bge-reranker-v2-m3 tách rõ doc liên quan: test "nghỉ phép" → 0.991 vs "VPN"/"mật khẩu" < 0.03. Trong pipeline rerank ~10 parent/query trên **CPU mất trung bình ~19 s/query** (chiếm ~88% latency query) — đúng như bài giảng: cross-encoder chính xác nhưng đắt, chỉ nên chạy trên top-N nhỏ. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Production: Faithfulness **0.892** (baseline 0.783), Answer Relevancy **0.841** (0.739), Context Precision **0.942** (0.925), Context Recall **0.933** (0.900). Phát hiện **answer_relevancy bị đo sai với tiếng Việt**: prompt gốc tiếng Anh sinh câu hỏi tiếng Anh rồi so cosine với câu hỏi tiếng Việt → câu trả lời đúng vẫn chỉ được 0.41. Adapt prompt sang tiếng Việt: cùng 20 answers **0.645 → 0.814**. |
| Contextual embeddings / Enrichment | M5 | `_enrich_single_call()` | Combined mode: **1 call/chunk** trả JSON {summary, questions, context, metadata}, 104 chunks enrich song song 4 luồng trong **~72 s**. Text được index = header tài liệu (tên + phiên bản + trạng thái "ĐÃ THAY THẾ") + context line + chunk + HyQA questions → retriever "nhìn thấy" phiên bản, câu hỏi user khớp với câu hỏi giả định. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

**1. Torch không load được trên Windows**
- *Exact error:* `OSError: [WinError 126] The specified module could not be found. Error loading "...\torch\lib\c10.dll" or one of its dependencies.`
- *Debug:* kiểm tra `C:\Windows\System32` → thiếu toàn bộ `vcruntime140.dll`, `vcruntime140_1.dll`, `msvcp140.dll`.
- *Fix:* cài Microsoft Visual C++ 2015–2022 Redistributable (x64). Đồng thời cài torch bản CPU (`--index-url .../whl/cpu`) vì máy không có GPU — tránh tải bản CUDA vài GB.

**2. Answer Relevancy thấp bất thường dù câu trả lời đúng**
- *Triệu chứng:* "Phụ cấp ăn trưa là 1.000.000 VNĐ/tháng" — đúng, ngắn gọn — chỉ được AR = 0.41.
- *Debug:* đọc cách tính metric: LLM sinh câu hỏi ngược từ answer (prompt tiếng Anh) → embedding cosine với câu hỏi gốc. Giả thuyết: lệch ngôn ngữ. Kiểm chứng bằng A/B trên **cùng 20 answers**: EN prompt 0.645 vs VI prompt 0.814 → giả thuyết đúng, lỗi ở thước đo chứ không ở pipeline.
- *Lỗi phát sinh khi fix:* `ValueError: llm must be either None or a BaseLanguageModel` — `ragas.adapt()` cần LangChain `ChatOpenAI` gốc, không nhận `LangchainLLMWrapper`.
- *Lỗi thứ hai:* adapt lúc runtime **không ổn định** — có lần `1 validation error for Prompt` (LLM dịch ra JSON lỗi) → fallback về prompt EN cho baseline nhưng production lại dùng VI → so sánh không công bằng. Ngoài ra ví dụ dịch xong bị bọc ```` ```json ```` khiến judge bắt chước → `Failed to parse output`.
- *Fix cuối:* dịch prompt **1 lần** (`scripts/build_ragas_prompts.py`, validate JSON + bỏ code fence), lưu `ragas_prompts/vietnamese/question_generation.json` vào repo; M4 chỉ load file này → deterministic, baseline & production chấm cùng một thước đo.

**3. `pip install -r requirements.txt` treo**
- *Triệu chứng:* >10 phút, `.venv/Lib/site-packages` chỉ có `pip`, process 56MB RAM.
- *Fix:* kill, cài torch CPU trước rồi mới cài phần còn lại với log (`--progress-bar off`) → resolver chạy xong ngay.

**4. Faithfulness thấp ở câu tính toán**
- Câu tạm ứng: answer "300.000 VNĐ" đúng nhưng Faithfulness 0.25 — judge không kiểm chứng được con số *suy ra* khi answer không nêu quy định gốc. Fix ở prompt: bắt buộc nêu quy định (kèm số liệu) làm căn cứ, quy đổi đơn vị thời gian (ngày ↔ tháng) trước khi tính.

**5. 2/3 PDF là scan ảnh**
- `BCTC.pdf` (thực chất là tờ khai thuế GTGT 01/GTGT) và Nghị định 13/2023 không có text layer → `load_documents()` bỏ qua. Không câu nào trong test set hỏi về 2 file này; muốn dùng cần OCR (Tesseract `vie` hoặc VLM).

**Kiến thức còn thiếu & cách bổ sung:** cách RAGAS tính từng metric bên trong (đọc source `ragas/metrics/_answer_relevance.py`, `ragas/llms/prompt.py`); đặc thù tokenizer tiếng Việt (đọc docs underthesea); chi phí cross-encoder trên CPU (đo bằng `benchmark_reranker()` + latency report).

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Trợ lý hỏi đáp tài liệu nội bộ tiếng Việt (quy chế, quy trình, văn bản pháp lý)

#### 1. Hiện trạng
- **Pipeline hiện tại:** chunk cố định theo số ký tự + dense search (1 embedding model) + top-k đưa thẳng vào LLM — giống `naive_baseline.py`.
- **Vấn đề / Bottlenecks:** (1) trả lời theo văn bản đã hết hiệu lực khi có nhiều phiên bản; (2) câu hỏi chứa mã số/con số chính xác ("Điều 13", "55 triệu") bị dense search bỏ lỡ; (3) nhiều văn bản là PDF scan; (4) chưa có bộ đánh giá định lượng.

#### 2. Kế hoạch cải tiến
1. **Chunking:** hierarchical (child ~256 ký tự để search, parent = 1 Điều/mục để trả về) kết hợp structure-aware cho văn bản có cấu trúc Chương/Điều/Khoản. Lý do: lab cho thấy parent-child giữ đủ ngữ cảnh cho câu multi-hop (Context Recall 0.933 so với baseline 0.900).
2. **Search:** Hybrid BM25 (underthesea) + bge-m3 + RRF. Thêm **metadata filter** theo `ngay_hieu_luc`/`trang_thai` để loại văn bản đã bị thay thế ngay từ bước retrieve.
3. **Reranking:** có — bge-reranker-v2-m3 nhưng chỉ trên top-10 **child** (ngắn) thay vì parent, chạy GPU hoặc ONNX; mục tiêu < 500 ms/query (lab: ~19 s/query trên CPU).
4. **Evaluation:** RAGAS 4 metrics với prompt **đã adapt tiếng Việt** (bài học từ lab) + custom metric "đúng phiên bản" (answer có trích phiên bản hiện hành không). Test set 50–100 câu chia theo loại: lookup, version, negation, multi-hop, numeric.
5. **Enrichment:** combined single-call (contextual prepend + HyQA + metadata) — rẻ (1 call/chunk) và giúp retriever thấy tên văn bản/phiên bản. OCR (Tesseract `vie`) cho PDF scan trước khi chunk.

#### 3. Timeline triển khai
- **Tuần 1:** OCR + structure-aware/hierarchical chunking cho toàn bộ văn bản; dựng test set 50 câu có ground truth.
- **Tuần 2:** Hybrid search + metadata filter theo hiệu lực; đo baseline vs hybrid bằng RAGAS (prompt tiếng Việt).
- **Tuần 3:** Reranker (GPU/ONNX) + enrichment combined; latency budget < 2 s/query end-to-end.
- **Tuần 4:** Failure analysis bottom-10 theo Error Tree, tinh chỉnh prompt; tích hợp CI chạy RAGAS mỗi lần đổi pipeline.
