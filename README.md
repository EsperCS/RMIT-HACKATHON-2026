# 🛡️ RMIT HACKATHON 2026: Multimodal Gatekeeper AI Safety Benchmark
### **Mô Hình Đa Phương Thức Qwen/Qwen3.5-2B Bảo Vệ Qwen/Qwen3.5-4B**

Hệ thống **Multimodal AI Safety Gatekeeper** phân loại nhị phân (`0 = Benign`, `1 = Jailbreak/Attack`) tiếp nhận đầu vào là cặp **(Image + Text)** hoặc **(Text-only)** mà **KHÔNG CẦN CHẠY OCR RỜI**. Hệ thống sử dụng mô hình nhỏ đa phương thức **`Qwen/Qwen3.5-2B`** làm Gatekeeper (kết hợp các đặc trưng *Explicit Reflexivity* theo arXiv:2510.01644, khả năng *Tự động lọc dữ liệu bị gán nhãn sai - Self-Filtering Poisoned Labels*, *QLoRA Fine-tuning* và *OOD Mahalanobis Monitor*) để bảo vệ mô hình lớn **`Qwen/Qwen3.5-4B`**.

Hệ thống hỗ trợ nạp dữ liệu tấn công liên tục theo từng đợt (**Continual Red-Teaming Drops**) từ người cộng sự qua thư mục `data/friend_drops/`, đồng thời tự động tải trực tiếp các bộ dữ liệu benchmark quốc tế chuẩn (**AdvBench**, **XSTest**, **Alpaca**) khi thư mục này chưa có dữ liệu. Toàn bộ mã nguồn đạt chuẩn mực công nghiệp (**Fault-Tolerant & Production-Ready**), chống tràn VRAM, tự động nối lại tiến trình khi mất kết nối (**Auto-Resume**), đồng bộ trực tiếp lên **Google Drive** và chạy mượt mà trên **Google Colab (GPU T4/L4)**.

---

## 🏛️ Kiến Trúc Hệ Thống (System Architecture)

```
                       [ Input Query: (Image + Text) hoặc (Text-only) ]
                                              │
                ┌─────────────────────────────┴─────────────────────────────┐
                ▼                                                           ▼
     [ Text De-obfuscation ]                                    [ Native Multimodal ]
  - Base64 / Hex / ROT13 / URL Decode                         - Qwen/Qwen3.5-2B
  - Phục hồi từ khóa rời rạc (b.o.m.b)                        - Token pixel patch [128, 384]*28*28
  - Vector 9 chiều Explicit Reflexivity (arXiv:2510.01644)    - Single-step Logits: σ(z_UNSAFE - z_SAFE)
                │                                             - Tự lọc nhãn bẩn (Self-Filtering Sanitizer)
                ▼                                             - QLoRA Multimodal Fine-tuning
     [ Tier 1: Fast Screen ]                                  - OOD Mahalanobis Anomaly Monitor
     (ReflexivityClassifier)                                                │
     Độ trễ: ~0.5 ms/query                                                 │
                │                                                           │
                └─────────────────────────────┬─────────────────────────────┘
                                              │
                                              ▼
                              [ Hybrid Defense Orchestrator ]
            ┌─────────────────────────────────┴─────────────────────────────────┐
            ▼                                                                   ▼
  [ 2-Tier Cascade Gatekeeper ]                               [ Multi-Signal Rank Ensemble ]
  - Text-only & (P < 0.10 hoặc P > 0.90)                      Kết hợp 4 tín hiệu an toàn bằng thứ hạng:
    -> Chốt quyết định ngay tại Tier 1 (~0.5ms)               P_Ens = w1*Rank(P_QLoRA) + w2*Rank(P_ZeroShot)
  - Có ảnh HOẶC rơi vào vùng xám [0.10, 0.90]                         + w3*Rank(P_OOD)   + w4*Rank(P_Reflex)
    -> Leo thang lên Tier 2: Deep Qwen3.5-2B                  Tối ưu trọng số bằng SLSQP (scipy.optimize)
                                              │
                                              ▼
                   [ Target Model: Qwen/Qwen3.5-4B + ProbGuard Lookahead ]
                   - Nếu P_Guard >= 0.50 -> Chặn ngay lập tức (Pre-check)
                   - Vùng xám [0.35, 0.50) -> Kích hoạt ProbGuard 10-Token Lookahead
                   - Cắt ngang phản hồi khi phát hiện token khởi đầu chấp thuận (Compliance Prefix)
```

---

## 📦 Cơ Chế Nạp Dữ Liệu & Phân Tách Tập Dữ Liệu Thi Đấu

### 1. Nạp Tập Dữ Liệu Cuộc Thi (`rmit-hackathon-2025` qua `kagglehub`)
Hệ thống sử dụng **duy nhất** bộ dữ liệu chính thức của cuộc thi được tải tự động qua `kagglehub`:
```python
import kagglehub

path = kagglehub.competition_download('rmit-hackathon-2025')
print("Path to competition files:", path)
```

### 2. Chiến Lược Phân Tách Dữ Liệu (Data Splitting Strategy)
Hệ thống tuân thủ nghiêm ngặt quy tắc an toàn bảo mật và kiểm thử:
- **Tập Train (3,500 mẫu)**: Dùng để phân chia thành 3 đợt Continual Red-Teaming Drops (Drop 1: 1,166 mẫu, Drop 2: 1,166 mẫu, Drop 3: 1,168 mẫu). Mỗi vòng được tiêm 10% nhãn độc giả lập (`noisy_label = 0`) để kiểm chứng năng lực của bộ lọc tự khử độc `Self-Filtering Poison Sanitizer`.
- **Tập Validation (500 mẫu)**: Dùng làm tập kiểm chứng độc lập (`data/processed/val.parquet`) cho việc hiệu chỉnh ngưỡng phát hiện và lựa chọn trọng số Ensemble.
- **Tập Test (Phần còn lại - Remainder)**: Dùng làm bộ kiểm thử chuẩn toàn cục (`data/processed/global_test.parquet`) để tính toán toàn bộ các bảng đối sánh (Bảng 0, 1, 2, 3 và đường cong ROC Curves).
- **TUYỆT ĐỐI KHÔNG ĐỤNG TỚI `test.csv`**: File `test.csv` của cuộc thi được giữ nguyên vẹn 100%, không bị nạp hay can thiệp vào quá trình huấn luyện và đánh giá benchmark nội bộ.

### 3. Đánh Giá Đa Phương Thức Native (Multimodal Visual Jailbreaks)
Khoảng 15% các mẫu tấn công được tự động kết xuất văn bản lên canvas ảnh (`Pillow ImageDraw`) để kiểm thử khả năng đọc hiểu thị giác tự nhiên của `Qwen/Qwen3.5-2B` mà **không cần chạy OCR rời**. Với các câu hỏi dạng text thuần, hệ thống tự động gắn kèm ảnh nền trắng chuẩn.

---

## 🚀 Hướng Dẫn Thực Thi Trên Google Colab (GPU T4/L4)

### Quy trình 1-Click qua Notebook `colab_runner.ipynb`:
1. Mở notebook `colab_runner.ipynb` trên Google Colab.
2. Chọn Runtime: **Runtime -> Change runtime type -> T4 GPU** (hoặc **L4 GPU**).
3. Chạy tuần tự các ô lệnh trong Notebook:
   - **Cell 1**: `!nvidia-smi` (Kiểm tra GPU).
   - **Cell 2**: `drive.mount('/content/drive')` (Bảo vệ dữ liệu, chống mất checkpoint khi ngắt kết nối).
   - **Cell 3**: Clone repo từ GitHub:
     ```bash
     !git clone https://github.com/EsperCS/RMIT-HACKATHON-2026.git
     %cd /content/RMIT-HACKATHON-2026
     !git pull origin main
     ```
   - **Cell 4**: `!bash setup_colab.sh` (Cài đặt các thư viện mới nhất gồm `transformers>=4.48.0`, `peft`, `qwen-vl-utils`, `bitsandbytes`, `accelerate`).
   - **Cell 5**: `!python -u run_experiment.py --quick_test` (**Chạy thử 2-3 phút**, kiểm tra thông luồng toàn bộ 4 bảng và biểu đồ ROC).
   - **Cell 6**: `!python -u run_experiment.py --full_run` (**Chạy thực nghiệm chính thức**).
   - **Cell 7**: Hiển thị trực tiếp 4 bảng Markdown/CSV và biểu đồ `results/roc_curves.png`.
   - **Cell 8**: Đẩy kết quả và báo cáo mới lên Git cho đồng đội.

---

## 📊 Ý Nghĩa Của 4 Bảng Kết Quả Thực Nghiệm

Toàn bộ kết quả được in trực tiếp ra terminal dưới dạng bảng Markdown chuẩn và tự động lưu thành các file `.csv` trong thư mục `results/`:

### 1. Bảng 0: `results/table0_poison_filtering.csv` (Khử Độc Nhãn Tự Động)
- Đo lường **Precision, Recall, F1** của cơ chế dùng chính Qwen3.5-2B để phát hiện các mẫu tấn công bị đối phương cố tình gán nhãn sai thành lành tính (`noisy_label = 0`).
- So sánh **ROC-AUC** của mô hình khi huấn luyện trên dữ liệu bẩn (Dirty) so với dữ liệu đã được làm sạch (Cleaned), minh chứng **ROC-AUC Delta (Mức tăng hiệu năng nhờ khử độc)**.

### 2. Bảng 1: `results/table1_method_comparison.csv` (So Sánh 5 Cấu Hình Phòng Thủ)
So sánh toàn diện 5 cấu hình với đầy đủ các chỉ số: `ROC_AUC`, `F1_Score`, `Recall (Tỷ lệ phát hiện tấn công)`, `Over-refusal (Tỷ lệ từ chối nhầm câu hỏi vô hại trên XSTest)`, `Latency (ms/query)`, `Peak VRAM (MB)`:
1. `1. Reflexivity Heuristic Only`: Bộ lọc quy tắc nhanh trích xuất đặc trưng phản thân (~0.5ms).
2. `2. Qwen3.5-2B Zero-shot`: Đánh giá Logits trực tiếp từ mô hình gốc không cần train.
3. `3. Qwen3.5-2B QLoRA (Dirty / Poisoned Data)`: Huấn luyện trên dữ liệu bị tiêm 10% nhãn độc.
4. `4. Qwen3.5-2B QLoRA (Cleaned Data)`: Huấn luyện sau khi đã lọc nhãn qua bộ Self-Filtering.
5. `5. Cascade & Multi-Signal Ensemble (Reflex + 2B + OOD)`: Kết hợp phân vị thứ hạng 4 tín hiệu an toàn và phễu phân tầng 2 lớp, cho độ chính xác cao nhất và tối ưu độ trễ.

### 3. Bảng 2: `results/table2_continual_and_novel.csv` (Tiến Hóa Dữ Liệu & Novel Attack)
- Đánh giá khả năng học liên tục qua `Continual Round 1`, `Round 2`, `Round 3` chứng minh mô hình nâng cao độ nhạy với các đòn tấn công mới mà **không bị tăng tỷ lệ từ chối nhầm (Over-refusal Rate)**.
- Giao thức **Leave-One-Drop-Out** kiểm thử khả năng tổng quát hóa đối với hình thức tấn công hoàn toàn mới chưa từng gặp trong tập huấn luyện.

### 4. Bảng 3: `results/table3_end_to_end_4b.csv` (Bảo Vệ Qwen3.5-4B Mục Tiêu)
- **Attack Success Rate (ASR) [càng thấp càng tốt]**: Tỷ lệ tấn công xuyên thủng qua mô hình mục tiêu.
- **Over-refusal Rate (FPR) [càng thấp càng tốt]**: Tỷ lệ từ chối nhầm câu hỏi an toàn.
- So sánh giữa:
  - `No Defense (Target 4B Only)`: Không có phòng thủ (ASR thường ~100%).
  - `Standard Guard (Qwen3.5-2B) + Target 4B`: Kiểm tra tiền thực thi.
  - `Cascade & ProbGuard 10-Token Lookahead + Target 4B`: Kết hợp phễu 2 tầng và cơ chế nhìn trước 10 token, chặn đứng mọi đòn tấn công Jailbreak tinh vi.

---

## 🛠️ Cấu Trúc Thư Mục Repository

```
RMIT-HACKATHON-2026/
├── README.md                  # Tài liệu hướng dẫn chi tiết tiếng Việt
├── .gitignore                 # Chặn checkpoints nặng, giữ kết quả CSV và .gitkeep
├── requirements.txt           # Danh sách thư viện tương thích Colab T4/L4 & Local
├── setup_colab.sh             # Script cài đặt siêu tốc trên Colab
├── config.py                  # Cấu hình tập trung cho Qwen3.5-2B & Qwen3.5-4B + Auto-Drive
├── colab_runner.ipynb         # Notebook 1-click điều phối trên Google Colab
├── data/
│   ├── friend_drops/          # Nơi cộng sự thả file drop mới (.csv, .jsonl, .parquet)
│   ├── downloads/             # Tự động tải & cache AdvBench, XSTest từ GitHub
│   ├── synthetic_images/      # Ảnh Typographic Jailbreak sinh tự động bằng Pillow
│   └── processed/             # Cache dữ liệu đã trộn & lọc độc (.parquet)
├── checkpoints/               # Lưu trọng số LoRA và file experiment_state.json (Auto-Resume)
├── results/                   # Lưu 4 bảng CSV và biểu đồ roc_curves.png
│   ├── table0_poison_filtering.csv
│   ├── table1_method_comparison.csv
│   ├── table2_continual_and_novel.csv
│   ├── table3_end_to_end_4b.csv
│   └── roc_curves.png
├── src/
│   ├── utils.py               # Seed, Anti-OOM, State Manager, Drive Sync, Path Resolver
│   ├── text_features.py       # Giải mã Base64/Hex/ROT13 + Explicit Reflexivity Classifier (0.5ms)
│   ├── data_mixer.py          # Quét friend_drops / tải online, sinh ảnh, cân bằng 50:25:25, tiêm độc
│   ├── qwen_vlm_branch.py     # Qwen3.5-2B Native VLM, Self-Filtering Sanitizer, QLoRA, OOD Monitor
│   ├── hybrid_defenses.py     # Multi-Signal Rank Ensemble, 2-Tier Cascade, ProbGuard Defender 4B
│   └── evaluator.py           # Tính toán chỉ số AI Safety, in Markdown và vẽ ROC Curves
└── run_experiment.py          # Master CLI điều phối toàn bộ pipeline
```

---

## 🧪 Các Tham Số Dòng Lệnh CLI

```bash
# Chạy nhanh 2 phút kiểm tra mã nguồn (phù hợp CPU và T4 test)
python -u run_experiment.py --quick_test

# Chạy thực nghiệm toàn diện trên toàn bộ dữ liệu
python -u run_experiment.py --full_run

# Chạy từng pha độc lập (Modular Execution)
python -u run_experiment.py --stage data     # Chỉ chuẩn bị và lọc dữ liệu
python -u run_experiment.py --stage guard    # Chạy nhánh Qwen3.5-2B Gatekeeper
python -u run_experiment.py --stage e2e      # Đánh giá bảo vệ Qwen3.5-4B
python -u run_experiment.py --stage eval     # Xuất lại các bảng kết quả và biểu đồ ROC

# Bật/tắt tự động nối tiếp tiến trình (mặc định bật)
python -u run_experiment.py --resume
python -u run_experiment.py --no_resume
```