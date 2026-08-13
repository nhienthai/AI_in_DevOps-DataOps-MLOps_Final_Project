# Kaggle Fine-Tuning Guide (XLM-RoBERTa + UIT-VSFC)

Hướng dẫn chạy huấn luyện mô hình **XLM-RoBERTa** trên bộ dữ liệu **UIT-VSFC** bằng Kaggle GPU (T4 / P100) miễn phí.

---

### 📌 LƯU Ý KHI CHẠY TRÊN KAGGLE NOTEBOOK
Mỗi bước bên dưới nên được chạy trong **từng Cell riêng biệt** để Kaggle Notebook nhận diện chính xác đường dẫn làm việc (`%cd`).

---

### Cell 1: Clone Repository & Di chuyển vào thư mục dự án

```python
import os
!git clone -b model/sentiment-training-setup https://github.com/nhienthai/AI_in_DevOps-DataOps-MLOps_Final_Project.git
%cd /kaggle/working/AI_in_DevOps-DataOps-MLOps_Final_Project
!pwd
```

---

### Cell 2: Cài đặt Thư viện

```python
!pip install -q -r requirements.txt
```

---

### Cell 3: Fine-tune XLM-RoBERTa trên GPU Kaggle

```python
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

### Cell 4: Kiểm tra Quality Gate & Benchmarking

```python
!python scripts/validate_model.py \
    --model-path ./artifacts/xlm-roberta \
    --model-type transformer \
    --min-macro-f1 0.85
```
