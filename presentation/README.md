# Hướng Dẫn Biên Dịch & Chỉnh Sửa Slide Thuyết Trình LaTeX (Beamer)

Tệp Slide: [`slides.tex`](slides.tex)  
Dự án: **RMIT Hackathon 2026 - Multimodal AI Safety Gatekeeper**  
Định dạng: **LaTeX Beamer (Khổ rộng 16:9, Madrid Clean Modern Palette)**

---

## 1. Cách xem và biên dịch Slide

### Cách 1: Sử dụng Overleaf (Khuyên Dùng - Nhanh Nhất)
1. Truy cập [Overleaf.com](https://www.overleaf.com/) và đăng nhập.
2. Chọn **New Project** $\to$ **Upload Project** (hoặc tạo Blank Project rồi dán toàn bộ nội dung của [`slides.tex`](slides.tex) vào).
3. Bấm **Recompile** (hoặc tổ hợp phím `Ctrl + Enter`).
4. Overleaf sẽ tạo ra file PDF trình chiếu sắc nét, sẵn sàng tải về.

### Cách 2: Biên dịch cục bộ (Local TeX Live / MiKTeX / VS Code)
Mở terminal tại thư mục `presentation/` và chạy:
```bash
pdflatex slides.tex
pdflatex slides.tex
```
*(Chạy 2 lần để cập nhật số trang và liên kết hyperref).*

---

## 2. Cấu trúc 12 Slide trong bài thuyết trình

1. **Slide 1 - Title Slide**: Tiêu đề nghiên cứu, tên nhóm và đơn vị tổ chức RMIT.
2. **Slide 2 - Motivation & Problem**: Bối cảnh tấn công Jailbreak (Prompt Injection, Roleplay, Typographic Attacks).
3. **Slide 3 - System Architecture (TikZ)**: Sơ đồ khối kiến trúc phân tầng 2-Tier Gatekeeper.
4. **Slide 4 - Data Engineering**: Phân chia không rò rỉ (3000 Train, 500 Val, 500 Test), tiêm nhãn độc 10% & sinh ảnh Typographic.
5. **Slide 5 - Self-Filtering Poison Sanitizer (Table 0)**: Công thức ngưỡng thích ứng $\tau = \mu + 1.2\sigma$ và Bảng 0 (Precision/Recall 100%).
6. **Slide 6 - Tier 1 Fast Screen**: Bộ giải mã De-obfuscator, 1,009 chiều đặc trưng (TF-IDF + Reflexivity), $0.5\text{ ms}$, $0\text{ MB}$ VRAM.
7. **Slide 7 - Tier 2 Deep VLM Guard**: Nén 4-bit NF4 Qwen3.5-2B QLoRA, đơn bước trích xuất logit và khoảng cách Mahalanobis OOD.
8. **Slide 8 - Nelder-Mead Optimization & Cascade**: Tối ưu trọng số trên tập Val, chốt chặn Non-Degradation Guard.
9. **Slide 9 - Benchmark Results (Table 1)**: Bảng đối sánh 5 phương pháp (Cascade Ensemble đạt ROC-AUC 1.0000, 106.4 ms).
10. **Slide 10 - Continual & Novel Attacks (Table 2)**: Kiểm chứng Continual Rounds 1, 2, 3 và tổng quát hóa tấn công mới Leave-One-Drop-Out.
11. **Slide 11 - Target Model 4B Defense (Table 3)**: Kỹ thuật 10-Token Lookahead Early Stopping (ASR kéo về 0.00%, độ trễ 125.3 ms).
12. **Slide 12 - Competition Deliverables & Summary**: File `submission.csv`, tính tái lập kỹ thuật và lời cảm ơn.

---

## 3. Tự động đồng bộ số liệu vào Slides (`slides.tex`)

Toàn bộ 4 bảng trong `slides.tex` đã được kết nối tự động với các file kết quả thực nghiệm trong thư mục `results/`.

### Cách 1: Tự động hoàn toàn sau khi chạy thực nghiệm
Khi chạy thực nghiệm xong:
```bash
python main.py --full_run
```
Hệ thống sẽ **tự động cập nhật trực tiếp** các con số mới nhất vào file `presentation/slides.tex`.

### Cách 2: Cập nhật thủ công bất cứ lúc nào
Nếu bạn đã có các file `results/table*.csv` và chỉ muốn đồng bộ vào slide:
```bash
python main.py --update_slides
```
Hoặc:
```bash
python presentation/update_slides.py
```
*(Nếu muốn cập nhật số liệu của tập Validation 500 mẫu: thêm cờ `--val`: `python main.py --update_slides --val`)*.

