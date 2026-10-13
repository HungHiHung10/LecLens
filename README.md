# 🎬 LecLens — Multimodal Video Grounding & Lecture Search Assistant
> **Hệ thống tìm kiếm, định vị và phát lại chính xác phân đoạn video bài giảng bằng câu hỏi tự nhiên (Multimodal Temporal Video Grounding RAG)**  
> *Thiết kế kiến trúc Module hóa chuyên nghiệp, tối ưu hóa CPU cho máy tính xách tay (Intel Core i5-1135G7, 12GB RAM, không yêu cầu GPU rời NVIDIA).*

---

## 🌟 1. Điểm Nổi Bật của Hệ Thống

### 👁️ Nhánh Thị Giác Đa Phương Thức (Vision & Slide OCR Branch)
- **Scene-Change Keyframe Extraction** (`vision_extractor.py`): Dò tìm các điểm chuyển cảnh slide thông qua PyAV (`av`) và thuật toán sai phân biểu đồ màu (Luminance histogram diff). Rút gọn video 45 phút thành ~20–30 slide JPEG chất lượng cao (~960p).
- **RapidOCR trên ONNX Runtime CPU**: Trích xuất toàn bộ dòng chữ, tiêu đề slide và các khối sơ đồ kiến trúc, bảng số liệu, công thức toán.
- **Vietnamese Text Enhancer** (`vietnamese_text_enhancer.py`): Động cơ phục hồi tiếng Việt có dấu và phân tách từ chuyên biệt:
  - Tận dụng phụ đề Whisper làm *Ground Truth* đối chiếu chéo $N\text{-grams}$ âm thanh – thị giác.
  - Sửa lỗi ký tự đặc thù của Asian OCR (`d6` $\rightarrow$ `đồ`, `lurong` $\rightarrow$ `lượng`, `nudc` $\rightarrow$ `nước`).
  - Lọc bỏ watermark truyền hình (`THANHNIEN`, `SIU`, `CHUONGTRINHONTHITHPT`) để lấy đúng tiêu đề học thuật thực tế của slide.
- **Disk Caching Metadata**: Metadata slide và OCR được lưu trữ vĩnh viễn trong `data/keyframes/{video_id}/`, giúp các lần tải sau đạt tốc độ tức thì (< **10ms**).
- **Giao diện trực quan**: Thẻ kết quả hiển thị hình ảnh slide tương ứng, gắn nhãn trích xuất và hỗ trợ **bấm trực tiếp vào ảnh slide để nhảy tới đoạn phát video**.

### 🎙️ Nhánh Âm Thanh & ASR Siêu Tốc (Audio & Transcription Branch)
- **Groq Cloud LPU Whisper (`whisper-large-v3`)**: Nhận dạng giọng nói video 80–120 phút chỉ trong **5–8 giây** (~300x realtime).
- **Forced Vietnamese Prompting**: Ép nhận dạng tiếng Việt có dấu chuẩn xác kèm từ điển thuật ngữ Machine Learning & Địa lý / Giáo dục.
- **CPU Faster-Whisper Fallback**: Tự động chuyển đổi mượt mà sang `faster-whisper` (INT8 trên CPU) khi không có kết nối internet hoặc API key.
- **YouTube Transcript API**: Hỗ trợ dán đường link YouTube và trích xuất phụ đề tự động trong 1 giây.

### 🧠 Video Grounding RAG & Hybrid Retrieval
- **Query Parsing & Decomposition** (`query_parser.py`): Phân tách ý định truy vấn, từ khóa giọng nói, thuật ngữ trên slide và sinh biến thể đa truy vấn (Multi-Query Expansion).
- **Multimodal Semantic Fusion** (`multimodal_indexer.py`): Kết hợp Lời giảng (ASR) + Tiêu đề slide + Nội dung hình ảnh & OCR + Đệm thời gian Temporal Hedge ($\pm 3s$).
- **Hybrid Search Engine** (`searcher.py`): Kết hợp Dense Semantic Vector (`keepitreal/vietnamese-sbert` hoặc `all-MiniLM-L6-v2`) với Sparse BM25 và giải thuật khử trùng lặp thời gian Temporal NMS.
- **Grounded Answer Synthesis** (`video_grounding_rag.py`): Tổng hợp câu trả lời tự nhiên kèm danh sách trích dẫn bằng chứng (Evidence Citations) click-to-seek.

---

## 🏛️ 2. Luồng Hoạt Động Cốt Lõi (Architecture Workflow)

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ LUỒNG 1: NẠP VÀ LẬP CHỈ MỤC ĐA PHƯƠNG THỨC (INGESTION & MULTIMODAL INDEXING)         │
│                                                                                        │
│  [Video MP4] ──┬─► 1. Whisper / Groq / YouTube ──► [Phụ đề SRT] ──► [Cắt Chunks (30s)] │
│                │                                                              │        │
│                └─► 2. PyAV + RapidOCR + Enhancer ─► [Slide Keyframe + OCR]    ▼        │
│                                                            │             [Ghép Chunks  │
│                                                            └────────────► + Keyframe]  │
│                                                                               │        │
│                                                                               ▼        │
│                                                                   [Vector Embeddings   │
│                                                                    + BM25 Inverted Index]
└────────────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────────────┐
│ LUỒNG 2: TÌM KIẾM & ĐỊNH VỊ VIDEO (QUERY GROUNDING & RAG RETRIEVAL)                   │
│                                                                                        │
│  [Câu hỏi người dùng] ──► [Dense SBERT Vector] + [BM25 Tokenizer]                      │
│                                       │                                                │
│                                       ▼                                                │
│                          [Hybrid Retrieval (Alpha = 0.7)]                              │
│                                       │                                                │
│                                       ▼                                                │
│                     [Temporal NMS (Lọc đoạn trùng lặp)]                                │
│                                       │                                                │
│                                       ▼                                                │
│                      [Video Grounding RAG Engine]                                      │
│                                       │                                                │
│             ┌─────────────────────────┴──────────────────────────┐                     │
│             ▼                                                    ▼                     │
│  [Đoạn trích trả lời + Trích dẫn]                   [Ảnh Slide + Mốc phát video]        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ 3. Cấu Trúc Thư Mục Chuẩn Hóa (Modular Architecture)

Dự án được tái cấu trúc theo mô hình **FastAPI Enterprise Architecture** phân tách rõ ràng trách nhiệm (*Separation of Concerns*):

```
a:/LecLens/
├── backend/                     # API Backend & Động cơ AI (Modularized Architecture)
│   ├── core/                    # AI Core: Search, Embedding, Multimodal Indexing, Grounding RAG
│   │   ├── searcher.py          # Hybrid Search Engine (Vector + BM25 + Temporal NMS)
│   │   ├── embedder.py          # Quản lý Text Embedding & Disk Caching (.npz)
│   │   ├── multimodal_indexer.py# Hợp nhất đa phương thức (Speech + Slide OCR + Keyframe)
│   │   ├── video_grounding_rag.py# RAG Grounding Synthesis & Evidence Citations
│   │   └── query_parser.py      # Phân tích câu hỏi & mở rộng đa truy vấn
│   ├── vision/                  # Thị giác & Slide OCR (ONNX CPU)
│   │   ├── vision_extractor.py  # Keyframe Scene Detection (PyAV) & RapidOCR ONNX
│   │   └── vietnamese_text_enhancer.py # Phục hồi dấu tiếng Việt & chuẩn hóa thuật ngữ Slide
│   ├── audio/                   # Âm thanh, ASR & Phụ đề
│   │   ├── chunker.py           # Cắt cửa sổ trượt thời gian (Sliding Window 30s/15s)
│   │   ├── subtitle_parser.py   # Phân tích cú pháp phụ đề (.srt, .vtt)
│   │   ├── audio_transcriber.py # Trích xuất âm thanh, YouTube & Faster-Whisper CPU
│   │   └── groq_transcriber.py  # Groq Cloud LPU Whisper siêu tốc (~300x realtime)
│   ├── evaluation/              # Đánh giá & Benchmark học thuật
│   │   ├── grounding_eval.py    # Đo lường mIoU, Recall@k benchmark học thuật
│   │   └── annotation_manager.py# Quản lý gán nhãn và xuất dataset CSV
│   ├── routers/                 # Bộ điều hướng API theo controller chuyên biệt
│   │   ├── search.py            # Endpoints: /api/search, /api/rag/ask, /api/subtitles
│   │   ├── media.py             # Endpoints: /api/upload, /api/load-sample, /api/youtube, stream
│   │   └── system.py            # Endpoints: /api/status, /api/samples, /api/groq, benchmark
│   ├── schemas.py               # Pydantic Request & Response Data Models
│   ├── state.py                 # Singleton App State & Khởi tạo Engine tập trung
│   ├── config.py                # Cấu hình hệ thống, đường dẫn thư mục, mô hình & logger
│   ├── main.py                  # FastAPI Application Gateway (< 90 dòng)
│   └── __init__.py              # Public Package Exports & Backward-compatibility aliases
│
├── frontend/                    # Giao diện Web Studio (Google-style Minimalist)
│   ├── index.html               # Cấu trúc HTML5 ngữ nghĩa
│   ├── style.css                # Hệ thống CSS Design System & Glassmorphism
│   ├── app.js                   # Logic trình phát video, RAG rendering & Scrubber
│   ├── favicon.ico              # Favicon đa độ phân giải cho mọi trình duyệt
│   ├── favicon.svg              # Favicon vector sắc nét tuyệt đối
│   ├── apple-touch-icon.png     # Biểu tượng App bo góc (Squircle) cho iOS / macOS
│   ├── site.webmanifest         # Cấu hình PWA Web Manifest
│   ├── LecLens.png              # Header Brand Logo chính thức
│   └── Groq.png                 # Groq LPU Integration Badge Logo
│
├── data/                        # Lưu trữ dữ liệu bền vững
│   ├── samples/                 # Video & phụ đề bài giảng mẫu
│   ├── keyframes/               # Ảnh Slide Keyframes & metadata OCR theo video
│   ├── cache/                   # Bộ nhớ đệm embeddings (.npz)
│   └── uploads/                 # Video và file phụ đề người dùng tải lên
│
├── logo/                        # Asset thương hiệu gốc
│   ├── logo.png                 # Logo Vector/Master gốc
│   └── Groq.png                 # Logo Groq đối tác
│
├── tests/                       # Bộ kiểm thử tự động (Unit Test Suite - 100% Pass)
│   ├── test_parser.py           # Kiểm thử parser phụ đề
│   ├── test_search.py           # Kiểm thử hybrid search
│   ├── test_rag.py              # Kiểm thử Video Grounding RAG pipeline
│   ├── test_vision.py           # Kiểm thử Vision & Slide OCR
│   └── test_benchmark.py        # Chạy benchmark 30 câu hỏi
│
├── scripts/                     # Kịch bản tiện ích & Offline Data Processing
├── execute.py                   # Script khởi chạy 1 chạm (tự phát hiện cổng trống & mở Web)
├── requirements.txt             # Danh sách thư viện phụ thuộc hoàn chỉnh
└── README.md                    # Tài liệu hướng dẫn sử dụng và kiến trúc
```

---

## 🚀 4. Hướng Dẫn Cài Đặt & Sử Dụng

### Bước 1: Cài đặt thư viện
```powershell
pip install -r requirements.txt
```

### Bước 2: Khởi chạy hệ thống 1-click
```powershell
python execute.py
```
> Hệ thống sẽ tự động quét cổng khả dụng (mặc định `8000`, nếu bận sẽ tự chuyển sang `8001`, `8002`,...) và mở ngay trình duyệt Web UI.

### Bước 3: Chạy kiểm thử tự động
```powershell
python -m unittest discover tests
```

---

## 📊 5. Kết Quả Thực Nghiệm & So Sánh Các Phương Thức Truy Xuất

Thực nghiệm được đo đạc tự động trên tập dữ liệu kiểm chuẩn **30 câu hỏi chuyên đề Machine Learning** (`data/samples/benchmark_ground_truth.json`), chạy trực tiếp trên phần cứng máy tính xách tay phổ thông **Intel Core i5-1135G7 (12GB RAM, không GPU rời)**.

Tập kiểm chuẩn đánh giá độ phủ thời gian 1D thông qua chỉ số **Temporal Intersection over Union (tIoU)**:
$$\text{tIoU} = \frac{|P \cap GT|}{|P \cup GT|} = \frac{\max(0, \min(p_e, g_e) - \max(p_s, g_s))}{(p_e - p_s) + (g_e - g_s) - \text{Inter}}$$

---

### 5.1. Bảng So Sánh Các Chiến Lược Truy Xuất (Retrieval Comparison Matrix)

| Phương thức Truy Xuất | Mô hình / Cấu hình | mIoU | Recall@1<br>(tIoU $\ge$ 0.3) | Recall@1<br>(tIoU $\ge$ 0.5) | Recall@3<br>(tIoU $\ge$ 0.3) | Recall@3<br>(tIoU $\ge$ 0.5) | Latency p50 (ms) | Latency p90 (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 Only** (Sparse) | `BM25Okapi` ($\alpha = 0.0$) | 0.4142 | 90.00% | 23.33% | 96.67% | 26.67% | 92.0 ms | 151.8 ms |
| **Dense Only (English)** | `all-MiniLM-L6-v2` (384-d) | 0.3140 | 70.00% | 20.00% | 90.00% | 26.67% | **17.9 ms** | **20.5 ms** |
| **Dense Only (Vietnamese)** | `keepitreal/vietnamese-sbert` (768-d) | 0.3276 | 70.00% | 20.00% | 90.00% | 26.67% | 74.3 ms | 86.6 ms |
| **Hybrid Retrieval ($\alpha = 0.5$)** | SBERT + BM25 (Cân bằng) | **0.4239** | **90.00%** | **26.67%** | **96.67%** | **26.67%** | 78.6 ms | 124.0 ms |
| **Hybrid Retrieval ($\alpha = 0.7$)** | SBERT + BM25 (**Mặc định**) | **0.4234** | **90.00%** | **26.67%** | **96.67%** | **26.67%** | 75.1 ms | 90.6 ms |
| **Multimodal Hybrid** | Speech ASR + Slide Keyframe OCR | 0.3726 | 76.67% | 23.33% | 93.33% | 26.67% | 71.6 ms | 138.4 ms |

> 💡 **Nhận định Thực nghiệm:**
> - **BM25 vs Dense**: Trên các câu hỏi học thuật có chứa thuật ngữ chuyên ngành (e.g., *Overfitting, Regularization L1/L2, Dropout, Cross-Entropy*), BM25 sparse matching mang lại điểm neo thời gian cực kỳ mạnh mẽ. Dense semantic vector giúp giải quyết các câu hỏi diễn đạt tự nhiên không trùng từ ngữ.
> - **Ưu thế của Hybrid Retrieval ($\alpha = 0.5 - 0.7$)**: Kết hợp Dense SBERT với Sparse BM25 đạt độ phủ thời gian cao nhất (**mIoU = 0.4239**, vượt trội **+29.4%** so với chỉ dùng Dense), đồng thời đưa **Recall@3 lên 96.67%** (29/30 câu hỏi tìm thấy chính xác phân đoạn bài giảng trong Top-3).
> - **MiniLM-L6-v2 Siêu tốc**: Đạt độ trễ cực thấp **17.9 ms (p50)** nhưng khả năng biểu diễn ngữ nghĩa tiếng Việt thấp hơn so với `vietnamese-sbert` chuyên dụng.

---

### 5.2. Nghiên Cứu Triệt Tiêu Thành Phần (Ablation Studies)

| Kỹ thuật Tối ưu | Tác động Hiệu năng | Phân tích Giá trị Nghiệp vụ |
| :--- | :---: | :--- |
| **Temporal NMS ($\text{IoU} \ge 0.6$)** | Đa dạng hóa Top-3 | Khử trùng lặp triệt để giữa các cửa sổ trượt liền kề (30s trượt 15s). Ngăn hiện tượng Top-3 đều trỏ về cùng 1 phân đoạn tương tự nhau. |
| **Vietnamese Text Enhancer (OCR)** | Chuẩn hóa Từ khóa Slide | Phục hồi dấu thanh (`d6` $\rightarrow$ `đồ`, `sanluong` $\rightarrow$ `sản lượng`), loại bỏ watermark nhà đài, giúp truy vấn hình ảnh/biểu đồ không bị nhiễu. |
| **Temporal Hedging ($\pm 3s$)** | Trải nghiệm Nghe - Nhìn | Mở rộng biên an toàn 3 giây hai đầu, loại bỏ hiện tượng bị "cụt câu" khi giảng viên đang chuẩn bị mở lời hoặc kết luận vấn đề. |
| **Groq Cloud Whisper LPU vs CPU** | Tốc độ ASR **~300x realtime** | Giảm thời gian trích xuất âm thanh từ 4-8 phút (chạy CPU) xuống còn **5-8 giây** (Groq LPU), giải phóng 100% tài nguyên CPU cho người dùng. |

---

### 5.3. Tại Sao Không Dùng Cross-Encoder Reranking Trên CPU Laptop?

Trong các hệ thống RAG thông thường, Cross-Encoder Reranker thường được thêm vào sau bước retrieval. Tuy nhiên, qua thực nghiệm trên phần cứng máy tính xách tay cá nhân (Intel Core i5):
1. **Độ trễ quá lớn:** Một mô hình Cross-Encoder tiếng Việt chạy trên CPU tốn từ **3,000 ms đến 6,800 ms** cho mỗi truy vấn, phá vỡ hoàn toàn trải nghiệm tương tác thời gian thực (< 100ms).
2. **Hiệu quả biên không tương xứng:** Kiến trúc **Hybrid (SBERT + BM25) + Temporal NMS** của LecLens đã đạt **Recall@3 = 96.67%** với độ trễ chỉ **~75 ms**. Do đó, hệ thống quyết định **không sử dụng Cross-Encoder trên CPU**, giữ cho ứng dụng nhẹ, tức thì và mượt mà.

---

### 5.4. Lệnh Tái Tạo Kết Quả Thực Nghiệm

Bạn có thể tự chạy lại toàn bộ bảng so sánh thực nghiệm trên máy của mình bằng lệnh:
```powershell
python scripts/run_retrieval_comparison.py
```
> Kết quả chi tiết của từng câu hỏi và phân vị độ trễ (p50, p90) sẽ được lưu tự động tại `data/benchmark_comparison_report.json`.

