# Kaggle Fine-Tuning Guide (XLM-RoBERTa + UIT-VSFC)

Hướng dẫn chạy huấn luyện mô hình **XLM-RoBERTa** trên bộ dữ liệu **UIT-VSFC** bằng Kaggle GPU (T4 / P100) miễn phí.

---

### Bước 1: Mở Notebook mới trên Kaggle
1. Đăng nhập vào [Kaggle](https://www.kaggle.com).
2. Tạo Notebook mới (New Notebook).
3. Chọn **Accelerator**: `GPU T4 x2` hoặc `GPU P100` trong mục *Notebook options*.

---

### Bước 2: Clone Git Repo và Cài đặt môi trường
Trong cell đầu tiên của Kaggle Notebook, chạy lệnh sau:

```bash
# 1. Clone repository
!git clone -b model/sentiment-training-setup https://github.com/nhienthai/AI_in_DevOps-DataOps-MLOps_Final_Project.git
%cd AI_in_DevOps-DataOps-MLOps_Final_Project

# 2. Cài đặt các thư viện cần thiết
!pip install -q -r requirements.txt
```

---

### Bước 3: Huấn luyện Baseline Model (TF-IDF + Logistic Regression)
Chạy thử nghiệm baseline nhanh để kiểm tra dữ liệu và lưu log MLflow:

```bash
!python scripts/train_model.py \
    --model-type baseline \
    --dataset tridm/UIT-VSFC \
    --output-dir ./artifacts
```

---

### Bước 4: Fine-tune mô hình XLM-RoBERTa trên GPU Kaggle
Chạy huấn luyện mô hình Transformer chính thức:

```bash
!python scripts/train_model.py \
    --model-type transformer \
    --model-name xlm-roberta-base \
    --dataset tridm/UIT-VSFC \
    --epochs 3 \
    --batch-size 16 \
    --lr 2e-5 \
    --output-dir ./artifacts/xlm-roberta
```

---

### Bước 5: Kiểm tra Quality Gate & Lưu Weights
Kiểm tra xem mô hình trained có đạt chỉ số Macro-F1 (≥ 0.85) và Latency budget hay không:

```bash
!python scripts/validate_model.py \
    --model-path ./artifacts/xlm-roberta \
    --model-type transformer \
    --min-macro-f1 0.85
```

Sau khi hoàn thành, bạn có thể nén thư mục `./artifacts/xlm-roberta` hoặc push weights lên HuggingFace Hub / MLflow Registry để phục vụ cho phần **Serving (FastAPI)** của dịch vụ.
