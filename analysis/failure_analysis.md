# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Phạm Cường Quốc
**Khóa:** K4 - Track 3A

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.7833 | **0.8921** | +0.1087 |
| Answer Relevancy | 0.7389 | **0.8408** | +0.1019 |
| Context Precision | 0.9250 | **0.9417** | +0.0167 |
| Context Recall | 0.9000 | **0.9333** | +0.0333 |

- **Naive baseline:** paragraph chunking (51 chunks) + dense-only bge-m3, top-3, prompt cơ bản.
- **Production:** hierarchical chunking (104 child / 26 parent) → M5 enrichment combined (1 call/chunk) → hybrid BM25 + bge-m3 + RRF trên child → map child → parent → bge-reranker-v2-m3 top-3 parent → gpt-4o-mini (temperature 0, prompt xử lý phiên bản & tính toán).
- Judge: gpt-4o-mini + text-embedding-3-small. `answer_relevancy` dùng prompt sinh câu hỏi **đã adapt sang tiếng Việt** (`ragas_prompts/vietnamese/`) cho **cả baseline và production** — prompt gốc tiếng Anh làm điểm thấp giả tạo (cùng 20 answers: 0.645 với prompt EN vs 0.814 với prompt VI).
- Corpus nhỏ, tài liệu ngắn (~1KB) nên baseline đã có retrieval tốt (CP 0.925 / CR 0.90). Cải thiện lớn nhất nằm ở **generation** (Faithfulness +0.11, Answer Relevancy +0.10): parent context đầy đủ + prompt yêu cầu nêu căn cứ.

### Latency breakdown (CPU, không GPU)

| Bước | Thời gian |
|------|-----------|
| [build] Chunking (26 tài liệu) | 0.07 s |
| [build] Enrichment M5 (104 chunks, 1 call/chunk, 4 luồng) | 71.9 s |
| [build] Indexing BM25 + bge-m3 → Qdrant | 134.9 s |
| [build] Load reranker | 16.5 s |
| [query] Hybrid search (avg / max) | 438 ms / 1121 ms |
| [query] Rerank cross-encoder (avg / max) | **19 253 ms** / 30 665 ms |
| [query] LLM generation (avg / max) | 2 155 ms / 3 424 ms |
| [query] **Tổng / query** (avg / max) | **21 845 ms** / 32 675 ms |

→ Rerank chiếm **~88%** latency query: cross-encoder 568M tham số chạy trên CPU với ~10 parent (~1KB) mỗi query. Chi tiết: `reports/latency_report.json`.

---

## Bottom-5 Failures

Xếp theo trung bình 4 metric (thấp nhất trước), từ `reports/ragas_report.json` → `failures`.

### #1 — avg 0.702
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày, phí 2%/tháng × 15.000.000 = 300.000 VNĐ/tháng → pro-rata ≈ 50.000 VNĐ cho 5 ngày.
- **Got:** "Nhân viên sẽ bị phạt **600.000 VNĐ**" — tính đúng 300.000/tháng và 50.000/5 ngày nhưng rồi **cộng hai số lại** (và cộng sai: 300.000 + 50.000 ≠ 600.000).
- **Worst metric:** Faithfulness = 0.29 (CR 0.67, CP 1.00, AR 0.86)
- **Error Tree:** Output sai → Context đúng? **Có** (top-1 là `tam_ung.md`, chứa "2%/tháng sau 15 ngày") → Query OK? **Có** → lỗi ở **Generation**: LLM suy luận số học sai.
- **Root cause:** gpt-4o-mini làm phép tính nhiều bước trong lúc viết câu trả lời → lẫn giữa phí/tháng và phí/5 ngày. Prompt yêu cầu "quy đổi đơn vị" giúp tách bước nhưng không đảm bảo bước tổng hợp cuối đúng.
- **Suggested fix:** Tách bước tính toán: cho LLM xuất biểu thức (`15_000_000 * 0.02 * 5/30`) → evaluate bằng Python (tool/function calling), hoặc self-check "kiểm tra lại kết quả cuối có khớp các bước trên không". Dùng model mạnh hơn cho câu có số liệu.

### #2 — avg 0.776
- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Có, bắt buộc theo v2.0 (email, VPN, hệ thống nội bộ). Chính sách cũ v1.0 không yêu cầu MFA.
- **Got:** "Có, tất cả nhân viên **bắt buộc** kích hoạt MFA cho email, VPN và các hệ thống nội bộ… phiên bản 2.0, hiệu lực 01/07/2024." — đúng.
- **Worst metric:** Context Recall = 0.50 (AR 0.60)
- **Error Tree:** Output đúng → Context đúng? **Có** (top-3 gồm cả `mat_khau_v2.md` và `mat_khau_v1.md`) → nhưng recall vẫn 0.5 → lỗi ở **thước đo / ground truth**.
- **Root cause:** Câu "v1.0 không yêu cầu MFA" là **claim về sự vắng mặt** — `mat_khau_v1.md` không nhắc đến MFA nên judge không tìm được câu nào trong context để "attribute" claim này → tính là thiếu. Answer Relevancy 0.60 vì answer thêm chi tiết (ngày hiệu lực) khiến câu hỏi sinh ngược không trùng hẳn câu gốc.
- **Suggested fix:** Không phải lỗi retrieval. Có thể (1) viết ground truth chỉ gồm claim có thể kiểm chứng từ văn bản, hoặc (2) prompt yêu cầu so sánh rõ với phiên bản cũ khi có cả hai phiên bản trong context ("v1.0 không có quy định MFA").

### #3 — avg 0.798
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Cam kết ≥ 1 năm; nghỉ sau 8 tháng là trước hạn → hoàn trả 100% = 25.000.000 VNĐ.
- **Got:** "Phải hoàn trả **100% chi phí**, tức **25.000.000 VNĐ**… cam kết ít nhất 1 năm… nghỉ sau 8 tháng, không đủ thời gian cam kết" — **đúng hoàn toàn**.
- **Worst metric:** Faithfulness = 0.50
- **Error Tree:** Output đúng → Context đúng? **Có** (`hoan_chi_dao_tao.md` top-1) → lỗi ở **thước đo (judge)**.
- **Root cause:** Faithfulness tách answer thành các statement rồi kiểm từng câu với context. Các statement chứa **dữ kiện từ câu hỏi** ("25 triệu", "nghỉ sau 8 tháng") không có trong context → judge chấm "không suy ra được" dù đó là input của user, không phải hallucination.
- **Suggested fix:** Về đánh giá: đưa câu hỏi vào ngữ cảnh khi chấm faithfulness, hoặc dùng custom metric cho câu numeric (so khớp đáp số). Về prompt: tách rõ "dữ kiện từ câu hỏi" và "quy định từ tài liệu" trong câu trả lời.

### #4 — avg 0.814
- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** 15 ngày (v2024 hiện hành); v2023 là 12 ngày nhưng đã bị thay thế.
- **Got:** "**15 ngày phép năm**… phiên bản 2024, hiệu lực 01/01/2024, tăng từ 12 ngày so với 2023." — đúng.
- **Worst metric:** Context Precision = 0.50
- **Error Tree:** Output đúng → Context đúng? **Có nhưng sai thứ tự**: reranker xếp `nghi_phep_nam_v2023.md` (đã thay thế) ở **#1**, v2024 ở #2 → lỗi ở **Reranking/ranking**.
- **Root cause:** Hai phiên bản gần như trùng nội dung → cross-encoder chỉ đo độ liên quan ngữ nghĩa, **không biết phiên bản nào còn hiệu lực**. LLM vẫn trả lời đúng nhờ header "Phiên bản/Ngày hiệu lực" trong parent + prompt ưu tiên bản mới nhất.
- **Suggested fix:** Metadata filter/boost theo `ngay_hieu_luc` và `trang_thai` (đã trích được trong header): loại hoặc hạ hạng tài liệu có "ĐÃ THAY THẾ" / phiên bản cũ hơn khi cùng chủ đề; hoặc cộng recency bonus vào rerank score.

### #5 — avg 0.816
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 + 3 (9÷3) = 18 ngày phép; lương Senior (P3-P4): 20–35 triệu VNĐ/tháng.
- **Got:** 18 ngày phép (đúng, có phép tính) — nhưng phần lương trả lời "được hưởng lương trong thời gian nghỉ phép năm, vì đây là phép có lương" → **hiểu sai ý "lương trong khoảng nào"**.
- **Worst metric:** Context Recall = 0.50
- **Error Tree:** Output sai một nửa → Context đúng? **Thiếu**: top-3 = nghỉ phép v2024, v2023, nghỉ phép không lương — **không có `bang_luong_2024.md`** → Query OK? **Không** — câu hỏi multi-hop 2 ý, cả query bị kéo về chủ đề "nghỉ phép" → lỗi ở **Retrieval (query)**.
- **Root cause:** Một query chứa 2 intent (ngày phép + khung lương). Embedding/BM25 của cả câu bị ý "nghỉ phép" lấn át; top-3 bị 2 phiên bản nghỉ phép chiếm chỗ → tài liệu lương rơi khỏi top-3. LLM không có dữ liệu lương nên "lái" sang nghĩa khác.
- **Suggested fix:** **Query decomposition**: tách thành sub-query ("số ngày phép năm thâm niên 9 năm", "khung lương Senior") → retrieve riêng → gộp context. Ngoài ra dedupe phiên bản cũ (fix #4) sẽ giải phóng 1 slot cho `bang_luong_2024.md`.

---

## Case Study (cho presentation)

**Question chọn phân tích:** #5 — "Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"

**Error Tree walkthrough:**
1. **Output đúng?** → Một nửa: 18 ngày phép đúng; khung lương bị bỏ qua và trả lời lạc đề ("phép có lương").
2. **Context đúng?** → Không đủ: thiếu `bang_luong_2024.md`; 2/3 slot là v2024 + v2023 của cùng chính sách nghỉ phép (Context Recall 0.5).
3. **Query rewrite OK?** → Không có query rewrite — query 2 intent đi nguyên vào hybrid search nên intent "nghỉ phép" chiếm ưu thế.
4. **Fix ở bước:** **Retrieval** — (a) query decomposition cho câu hỏi nhiều ý; (b) metadata filter loại phiên bản đã bị thay thế để không chiếm slot top-k; (c) prompt: nếu context không có thông tin cho một ý thì nói rõ "không tìm thấy" thay vì diễn giải lại câu hỏi.

**Nếu có thêm 1 giờ, sẽ optimize:**
- **Metadata filter theo phiên bản** (fix #4, #5): parse `Phiên bản / Ngày hiệu lực / Trạng thái` từ header → loại tài liệu "ĐÃ THAY THẾ" khỏi top-k khi đã có bản mới cùng chủ đề.
- **Query decomposition** cho câu multi-hop (fix #5).
- **Tính toán bằng code** thay vì để LLM tự cộng trừ (fix #1).
- **Giảm latency rerank** (~19 s → < 1 s): rerank trên **child** (ngắn) thay vì parent, chỉ top-10, hoặc chạy ONNX/GPU.
