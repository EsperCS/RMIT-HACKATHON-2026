# 🛡️ RMIT HACKATHON 2026: Multimodal Gatekeeper AI Safety Benchmark

Hệ thống **Multimodal AI Safety Gatekeeper** phân loại nhị phân (`0 = Benign`, `1 = Jailbreak/Attack`) kết hợp giữa nhánh **Fast Multilingual Encoder** (XLM-RoBERTa + Explicit Reflexivity Features) và nhánh **Native Vision-Language Model** (Qwen2-VL / Qwen3.5-2B) có khả năng **Tự động lọc dữ liệu bị gán nhãn sai (Self-Filtering Poisoned Labels)** để bảo vệ mô hình ngôn ngữ lớn mục tiêu (Target Model: Qwen2.5-3B / Qwen3.5-4B).

Toàn bộ dữ liệu được **tải tự động từ các bộ benchmark trực tuyến chuẩn quốc tế** (AdvBench, XSTest, Alpaca) và lưu vào `data/downloads/`, không yêu cầu nạp file thủ công. Hệ thống được thiết kế chuẩn chuẩn mực công nghiệp (**Fault-Tolerant & Production-Ready**), hỗ trợ đồng bộ qua Git và thực thi trực tiếp trên **Google Colab (GPU T4/L4)**.

---

## 🏛️ Kiến Trúc Hệ Thống (System Architecture)

```
                       [ Input Query: (Image + Text) or (Text-only) ]
                                             │
               ┌─────────────────────────────┴─────────────────────────────┐
               ▼                                                           ▼
     [ Text De-obfuscation ]                                    [ Native Multimodal ]
  - Base64 / Hex / ROT13 Decoding                             - Qwen2-VL / Qwen3.5-2B
  - Dotted-word reconstruction (b.o.m.b)                      - Zero-shot Logits Extraction
  - Explicit Reflexivity Vector (arXiv:2510.01644)             - Self-Filtering Poison Cleaner
               │                                              - QLoRA Multimodal Adapter
               ▼                                              - Mahalanobis OOD Monitor
   [ Fast Encoder Branch ]                                                 │
   (XLM-R / DeBERTa + MLP)                                                 │
   Latency: ~2.5 ms/query                                                  │
               │                                                           │
               └─────────────────────────────┬─────────────────────────────┘
                                             │
                                             ▼
                             [ Hybrid Defense Orchestrator ]
           ┌─────────────────────────────────┴─────────────────────────────────┐
           ▼                                                                   ▼
 [ Cascade Gatekeeper (2-Tier Funnel) ]                      [ Rank-Averaging Ensemble ]
 - Text-only & P < 0.15 hoặc P > 0.85 -> Chốt ngay (~3ms)    P_Ens = w1*Rank(P_Enc) + w2*Rank(P_VLM)
 - Có ảnh HOẶC vùng xám [0.15, 0.85] -> Đẩy sang VLM         Tối ưu hóa weights bằng scipy.optimize
                                             │
                                             ▼
                     [ Target Model (Qwen-3B/4B) + ProbGuard ]
                     - Nếu P_Guard >= 0.50 -> Chặn ngay lập tức
                     - Vùng xám [0.40, 0.70] -> 10-Token Lookahead
                     - Cắt ngang khi phát hiện compliance prefix
```

---

## 📦 Nguồn Dữ Liệu Benchmark Trực Tuyến (Online Datasets)

Hệ thống tự động tải trực tiếp các bộ dữ liệu từ internet khi chạy `run_experiment.py`, lưu và cache tại `data/downloads/`:

| Dataset | Nguồn | Mục đích trong Pipeline |
| :--- | :--- | :--- |
| **AdvBench** | `llm-attacks/llm-attacks` (GitHub Raw) | Tập 520 câu tấn công Adversarial/Jailbreak thực tế, chia thành 3 đợt huấn luyện tích lũy (Continual Rounds) |
| **XSTest** | `paul-rottger/xstest` (GitHub Raw) | Tập 450 câu **Hard Benign** (chứa từ khóa nhạy cảm giả như *"kill a process"*) để đo lường và triệt tiêu **Over-refusal Rate** |
| **Alpaca Cleaned** | Multilingual (Anh / Việt) | Tập câu hỏi thường nhật (**Normal Benign**) đảm bảo mô hình giữ nguyên độ hữu dụng thông thường |
| **Synthetic Typographic** | Tự sinh bằng `Pillow (ImageDraw)` | Kết xuất các văn bản độc hại thành hình ảnh nhằm kiểm thử VLM Native trực tiếp không qua OCR rời |

Ngoài ra, hệ thống áp dụng kỹ thuật **Disguised Benign Pairing** (bọc 20% câu Benign vào khung nhập vai hoặc thẻ học tập mà vẫn giữ `label = 0`) để chống học vẹt cấu trúc prompt.

---

## 🚀 Danh Sách File Cần Chạy Theo Thứ Tự Trên Colab

Để chạy thực nghiệm từ đầu đến cuối trên Google Colab và đồng bộ qua lại qua Git:

| Bước | Lệnh / File thực thi | Mục đích | Thời gian ước tính |
| :--- | :--- | :--- | :--- |
| **1** | Mở file `colab_runner.ipynb` trên Google Colab | Nền tảng điều phối 1-click trên trình duyệt | 10 giây |
| **2** | Chọn Runtime: **GPU T4** hoặc **L4** | Cấp phát GPU có CUDA | 10 giây |
| **3** | `from google.colab import drive; drive.mount('/content/drive')` | Mount Drive để tự động backup checkpoints & results | 15 giây |
| **4** | `!git clone https://github.com/EsperCS/RMIT-HACKATHON-2026.git`<br>`%cd RMIT-HACKATHON-2026`<br>`!git pull origin main` | Tải mã nguồn mới nhất từ GitHub | 15 giây |
| **5** | `!bash setup_colab.sh` | Cài đặt siêu tốc các thư viện cần thiết (không phá vỡ torch sẵn có của Colab) | ~60 giây |
| **6** | `!python -u run_experiment.py --quick_test` | **Kiểm tra thông luồng toàn bộ pipeline**: tải online datasets, sinh ảnh mẫu, lọc độc, train và xuất đủ 4 bảng kết quả | **~2-3 phút** |
| **7** | `!python -u run_experiment.py --full_run` | **Chạy thực nghiệm chính thức** trên toàn bộ dữ liệu tải về | ~15-25 phút |
| **8** | Xem 4 bảng CSV và biểu đồ `results/roc_curves.png` | Phân tích và đánh giá kết quả | Tức thì |
| **9** | `!git add results/*.csv results/*.png`<br>`!git commit -m "Update benchmark results"`<br>`!git push origin main` | Đẩy kết quả về lại GitHub cho đồng đội xem | 10 giây |

---

## 📊 Ý Nghĩa Của 4 Bảng Kết Quả Thực Nghiệm

Toàn bộ kết quả được tự động in ra màn hình Terminal dưới dạng Markdown và xuất thành file `.csv` trong thư mục `results/`:

### 1. Bảng 0: `results/table0_poison_filtering.csv` (Self-Filtering Sanitizer)
- **Precision / Recall / F1**: Đo lường độ chính xác của cơ chế dùng chính VLM để truy quét các mẫu thực chất là Jailbreak nhưng bị cố tình đầu độc lật nhãn thành Benign (`noisy_label = 0`).
- **ROC-AUC Delta**: Chứng minh mức tăng vọt về độ chính xác khi mô hình được huấn luyện trên tập dữ liệu đã qua khử độc so với dữ liệu bẩn.

### 2. Bảng 1: `results/table1_method_comparison.csv` (So Sánh Các Phương Pháp)
So sánh 5 cấu hình phòng thủ:
1. `Fast Encoder Only`: Tốc độ cực nhanh (~2.5ms), nhẹ VRAM.
2. `VLM Guard (Zero-shot Logits)`: Không cần huấn luyện, bắt tốt các tấn công đa phương thức kết hợp ảnh.
3. `VLM Guard (QLoRA Cleaned)`: Huấn luyện thích nghi miền trên dữ liệu đã làm sạch.
4. `Cascade Gatekeeper (Encoder -> VLM)`: Phễu 2 tầng kết hợp tốc độ của Encoder và chiều sâu của VLM.
5. `Ensemble (Encoder + VLM Rank-Avg)`: Đạt ROC-AUC cao nhất bằng cách kết hợp phân vị thứ hạng.

### 3. Bảng 2: `results/table2_continual_and_novel.csv` (Tiến Hóa Liên Tục & Novel Attack)
- Theo dõi sự cải thiện của mô hình qua `Round 1` -> `Round 2` -> `Round 3` chứng minh mô hình ngày càng mạnh hơn khi tích lũy dữ liệu mà **không bị tăng tỷ lệ từ chối nhầm (Over-refusal Rate)**.
- Đánh giá khả năng bắt các đòn tấn công chưa từng gặp qua giao thức `Leave-One-Drop-Out`.

### 4. Bảng 3: `results/table3_end_to_end_target.csv` (Bảo Vệ Qwen-3B/4B)
- **Attack Success Rate (ASR) [lower is better]**: Tỷ lệ tấn công vượt rào thành công (càng thấp càng an toàn).
- **Over-refusal Rate (FPR) [lower is better]**: Tỷ lệ mô hình từ chối nhầm các câu hỏi vô hại (Hard Benign từ XSTest).
- **10-Token Lookahead (ProbGuard)**: Cắt đứt các trường hợp mô hình mục tiêu bắt đầu đồng ý trả lời câu hỏi nguy hại trước khi sinh toàn bộ văn bản.

---

## 🛠️ Cấu Trúc Mã Nguồn

```
RMIT-HACKATHON-2026/
├── README.md                  # Hướng dẫn chi tiết bằng tiếng Việt
├── .gitignore                 # Chặn checkpoints nặng, giữ kết quả CSV và .gitkeep
├── requirements.txt           # Danh sách thư viện tương thích Colab T4/L4
├── setup_colab.sh             # Script cài đặt siêu tốc trên Colab
├── config.py                  # Cấu hình tập trung, Auto-Drive, VLM pixel caps
├── colab_runner.ipynb         # Notebook 1-click chạy thực nghiệm
├── data/
│   ├── downloads/             # Tự động tải & cache AdvBench, XSTest từ GitHub
│   ├── synthetic_images/      # Ảnh Typographic Jailbreak sinh tự động bằng Pillow
│   └── processed/             # Cache dữ liệu đã trộn & lọc độc (.parquet)
├── checkpoints/               # Lưu trọng số LoRA và experiment_state.json
├── results/                   # Bảng kết quả CSV và biểu đồ roc_curves.png
├── src/
│   ├── utils.py               # Seed, Anti-OOM, State Manager, Drive Sync, Path Resolver
│   ├── text_features.py       # Giải mã Base64/Hex/ROT13 + Reflexivity Features
│   ├── data_mixer.py          # Tải AdvBench/XSTest, trộn ảnh+chữ, Disguised Benign, tiêm nhãn độc
│   ├── encoder_branch.py      # Nhánh Fast Encoder (XLM-R + Reflexivity)
│   ├── qwen_vlm_branch.py     # Nhánh Native VLM (Zero-shot, Sanitizer, QLoRA)
│   ├── hybrid_defenses.py     # Cascade, Ensemble và ProbGuard Lookahead
│   └── evaluator.py           # Tính toán chỉ số, xuất bảng và vẽ ROC
└── run_experiment.py          # Master CLI điều phối
```

---

## 🧪 Các Tham Số CLI Nâng Cao

```bash
# Chạy nhanh 2 phút kiểm tra mã nguồn
python -u run_experiment.py --quick_test

# Chạy toàn bộ dữ liệu
python -u run_experiment.py --full_run

# Chạy riêng lẻ từng pha (hữu ích khi muốn cô lập kiểm tra)
python -u run_experiment.py --stage data     # Chỉ tải và chuẩn bị dữ liệu
python -u run_experiment.py --stage encoder  # Chỉ chạy nhánh Fast Encoder
python -u run_experiment.py --stage vlm      # Chỉ chạy nhánh VLM & lọc độc
python -u run_experiment.py --stage eval     # Chỉ xuất lại các bảng kết quả

# Tự động chạy tiếp từ bước bị ngắt
python -u run_experiment.py --full_run --resume

# Chỉ định thư mục Google Drive để lưu bản sao lưu
python -u run_experiment.py --drive_backup_dir "/content/drive/MyDrive/MyHackathonBackup"
```