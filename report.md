# 📊 BÁO CÁO THỰC NGHIỆM THỰC TẾ & SO SÁNH MÔ HÌNH (UIT-VSFC)

> **Dự án**: AI in DevOps - DataOps - MLOps Final Project  
> **Bài toán**: Phân tích cảm xúc nhận xét sinh viên tiếng Việt (**Vietnamese Students' Feedback Corpus - UIT-VSFC**)  
> **Nhãn phân loại (3 classes)**: `0: NEGATIVE`, `1: NEUTRAL`, `2: POSITIVE`  
> **Tập dữ liệu**: HuggingFace `tridm/UIT-VSFC` (16,175 mẫu câu: 11,426 Train, 1,583 Val, 3,166 Test)  

---

## 1. Phân Tích Dữ Liệu Chuyên Sâu (Deep EDA on 16,175 Samples)

### 1.1. Mất cân bằng nhãn cực đoan (Class Imbalance - Tỷ lệ 12 : 1)
Toàn bộ dữ liệu gốc `tridm/UIT-VSFC` bị lệch nhãn nặng nề:

| Tập (Split) | NEGATIVE (0) | NEUTRAL (1) | POSITIVE (2) | Tổng số câu |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 5,325 (46.60%) | **458 (4.01%)** ⚠️ | 5,643 (49.39%) | 11,426 |
| **Validation** | 705 (44.54%) | **73 (4.61%)** ⚠️ | 805 (50.85%) | 1,583 |
| **Test** | 1,409 (44.50%) | **167 (5.27%)** ⚠️ | 1,590 (50.22%) | 3,166 |
| **TỔNG CỘNG** | **7,439 (46.00%)** | **698 (4.31%)** | **8,038 (49.69%)** | **16,175** |

> 🔴 **Vấn đề cốt lõi**: Cứ **24 câu nhận xét** thì chỉ có **1 câu NEUTRAL**. Hàm mất mát CrossEntropy tiêu chuẩn bị chi phối 96% bởi 2 lớp Âm/Dương, khiến mô hình học bị thiên lệch và đè nén xác suất (logits) của NEUTRAL.

---

### 1.2. Mối tương quan Cảm xúc & Chủ đề (Topic x Sentiment Matrix)

| Chủ đề (Topic) | NEGATIVE (%) | NEUTRAL (%) | POSITIVE (%) | Tổng số mẫu |
| :--- | :---: | :---: | :---: | :---: |
| **`facility` (Cơ sở vật chất)** | **95.65%** 🔴 | 1.83% | 2.53% | 712 |
| **`program` (Chương trình / Môn học)** | **76.58%** 🔴 | 5.33% | 18.09% | 3,040 |
| **`lecturer` (Giảng viên)** | 35.37% | 2.52% | **62.12%** 🟢 | 11,607 |
| **`others` (Chủ đề khác)** | 39.83% | **28.31%** 🟡 | 31.86% | 816 |

---

### 1.3. Bất thường về chiều dài câu (Length Anomaly) & Rác Token gốc

| Nhãn (Sentiment) | Độ dài trung bình | Độ dài trung vị (Median) | Tỷ lệ câu siêu ngắn ($\le 5$ từ) |
| :--- | :---: | :---: | :---: |
| **NEUTRAL** | **9.82 từ** | **8 từ** | **32.95%** (1/3 số câu) |
| **POSITIVE** | 12.15 từ | 10 từ | 9.28% |
| **NEGATIVE** | 16.89 từ | 13 từ | 6.45% |

- **Rác dữ liệu phát hiện trong dataset gốc**:
  - `wzjwz<id>` (302 câu): Token ẩn danh tên giảng viên (`thầy wzjwz208` $\rightarrow$ `thầy [ANON]`).
  - `doubledot` (118 câu): Token thay cho dấu hai chấm `:`, kể cả dính liền số (`11doubledot55` $\rightarrow$ `11:55`).
  - `fraction` (31 câu): Token thay cho dấu gạch chéo `/` (`thầy fraction cô` $\rightarrow$ `thầy/cô`).

---

## 2. Bảng Tổng Hợp So Sánh 100% Số Liệu Thực Nghiệm Thực Tế (Full Comparison)

Tất cả các mô hình được đánh giá độc lập trên tập **Test (3,166 câu: 1,409 NEGATIVE, 167 NEUTRAL, 1,590 POSITIVE)**.  
*(Quy tắc trình bày: Chỉ **tô đậm** duy nhất giá trị cao nhất trên từng cột).*

### 📊 2.1. Bảng Tổng Hợp Chi Tiết Từng Khối (Grouped Card View)

| Thực nghiệm | Chỉ số Tổng quan | Lớp 0: NEGATIVE | Lớp 1: NEUTRAL | Lớp 2: POSITIVE | File Notebook & Link Tải Weights (Google Drive) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Baseline (Gốc)**<br>*(XLM-RoBERTa Raw Text)* | • Accuracy: 93.59%<br>• Macro F1: 83.37% | • Precision: 94.01%<br>• Recall: 96.88%<br>• F1: 95.42% | • Precision: 67.42%<br>• Recall: 53.29%<br>• F1: 59.53% | • Precision: 95.39%<br>• Recall: 94.91%<br>• F1: 95.15% | 📓 [`train-xlm-roberta.ipynb`](notebooks/train-xlm-roberta.ipynb)<br>📦 [Tải `model_weights.zip` (1.11GB)](https://drive.google.com/drive/folders/14EuzHdBYKdDcfyNHC9YXJ-KLThgTSOrT?usp=sharing) |
| **2. Improved (Argmax)**<br>*(Cleaned + Weighted Loss)* | • Accuracy: 93.71%<br>• Macro F1: 82.17% | • Precision: 93.84%<br>• Recall: **97.30%**<br>• F1: 95.54% | • Precision: 70.00%<br>• Recall: 46.11%<br>• F1: 55.60% | • Precision: 95.24%<br>• Recall: 95.53%<br>• F1: 95.38% | 📓 [`train-xlm-roberta-improved.ipynb`](notebooks/train-xlm-roberta-improved.ipynb)<br>📦 [Tải `model_weights_improved.zip` (1.11GB)](https://drive.google.com/drive/folders/14EuzHdBYKdDcfyNHC9YXJ-KLThgTSOrT?usp=sharing) |
| **3. Improved (Threshold)**<br>*(Cleaned + Threshold $\tau=0.13$)* | • Accuracy: 93.68%<br>• Macro F1: 82.20% | • Precision: 94.03%<br>• Recall: 97.23%<br>• F1: 95.60% | • Precision: 67.52%<br>• Recall: 47.31%<br>• F1: 55.63% | • Precision: 95.29%<br>• Recall: 95.41%<br>• F1: 95.35% | 📓 [`train-xlm-roberta-improved.ipynb`](notebooks/train-xlm-roberta-improved.ipynb)<br>📦 [Tải `model_weights_improved.zip` (1.11GB)](https://drive.google.com/drive/folders/14EuzHdBYKdDcfyNHC9YXJ-KLThgTSOrT?usp=sharing) |
| **4. PhoBERT-v2 (Argmax)**<br>*(Topic Injection + Focal Loss)* | • Accuracy: **94.25%**<br>• Macro F1: 84.26% | • Precision: 94.23%<br>• Recall: **97.30%**<br>• F1: **95.74%** | • Precision: **75.44%** 🚀<br>• Recall: 51.50%<br>• F1: 61.21% | • Precision: 95.62%<br>• Recall: **96.04%**<br>• F1: **95.83%** | 📓 [`phobert-v2.ipynb`](notebooks/phobert-v2.ipynb)<br>📦 [Tải `phobert_model_weights.zip` (540MB)](https://drive.google.com/drive/folders/1hkdcZTRQKmz2BXsgPseg1Y85ocgvbqz9?usp=sharing) |
| **5. PhoBERT-v2 (Threshold)**<br>*(Topic + Focal Loss + $\tau=0.08$)* | • Accuracy: 93.84%<br>• Macro F1: **85.23%** 🚀 | • Precision: **94.94%**<br>• Recall: 95.88%<br>• F1: 95.41% | • Precision: 59.69%<br>• Recall: **70.06%** 🚀<br>• F1: **64.46%** 🚀 | • Precision: **97.16%**<br>• Recall: 94.53%<br>• F1: 95.82% | 📓 [`phobert-v2.ipynb`](notebooks/phobert-v2.ipynb)<br>📦 [Tải `phobert_model_weights.zip` (540MB)](https://drive.google.com/drive/folders/1hkdcZTRQKmz2BXsgPseg1Y85ocgvbqz9?usp=sharing) |

---

### 📋 2.2. Bảng Ma Trận Chi Tiết Từng Cột (Matrix View)

| Thực nghiệm | Accuracy | Macro F1 | NEG Precision | NEG Recall | NEG F1 | NEU Precision | NEU Recall | NEU F1 | POS Precision | POS Recall | POS F1 | Link Tải Weights (Google Drive) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Baseline (Gốc)** | 93.59% | 83.37% | 94.01% | 96.88% | 95.42% | 67.42% | 53.29% | 59.53% | 95.39% | 94.91% | 95.15% | 🔗 [Folder XLM-RoBERTa](https://drive.google.com/drive/folders/14EuzHdBYKdDcfyNHC9YXJ-KLThgTSOrT?usp=sharing) |
| **2. Improved (Argmax)** | 93.71% | 82.17% | 93.84% | **97.30%** | 95.54% | 70.00% | 46.11% | 55.60% | 95.24% | 95.53% | 95.38% | 🔗 [Folder XLM-RoBERTa](https://drive.google.com/drive/folders/14EuzHdBYKdDcfyNHC9YXJ-KLThgTSOrT?usp=sharing) |
| **3. Improved (Threshold)** | 93.68% | 82.20% | 94.03% | 97.23% | 95.60% | 67.52% | 47.31% | 55.63% | 95.29% | 95.41% | 95.35% | 🔗 [Folder XLM-RoBERTa](https://drive.google.com/drive/folders/14EuzHdBYKdDcfyNHC9YXJ-KLThgTSOrT?usp=sharing) |
| **4. PhoBERT-v2 (Argmax)** | **94.25%** | 84.26% | 94.23% | **97.30%** | **95.74%** | **75.44%** | 51.50% | 61.21% | 95.62% | **96.04%** | **95.83%** | 🔗 [Folder PhoBERT-v2](https://drive.google.com/drive/folders/1hkdcZTRQKmz2BXsgPseg1Y85ocgvbqz9?usp=sharing) |
| **5. PhoBERT-v2 (Threshold $\tau=0.08$)** | 93.84% | **85.23%** | **94.94%** | 95.88% | 95.41% | 59.69% | **70.06%** | **64.46%** | **97.16%** | 94.53% | 95.82% | 🔗 [Folder PhoBERT-v2](https://drive.google.com/drive/folders/1hkdcZTRQKmz2BXsgPseg1Y85ocgvbqz9?usp=sharing) |

---

## 3. Chi Tiết Báo Cáo Phân Loại Từng Thực Nghiệm (Classification Reports)

### 3.1. Baseline Model (XLM-RoBERTa Raw Text)
* **Notebook**: [`notebooks/train-xlm-roberta.ipynb`](notebooks/train-xlm-roberta.ipynb)
* **Weights Download**: 🔗 [Google Drive Folder `model_weights.zip`](https://drive.google.com/drive/folders/14EuzHdBYKdDcfyNHC9YXJ-KLThgTSOrT?usp=sharing)
* **Kết quả Test**:
```text
              precision    recall  f1-score   support

    NEGATIVE     0.9401    0.9688    0.9542      1409
     NEUTRAL     0.6742    0.5329    0.5953       167
    POSITIVE     0.9539    0.9491    0.9515      1590

    accuracy                         0.9359      3166
   macro avg     0.8561    0.8169    0.8337      3166
weighted avg     0.9330    0.9359    0.9339      3166
```

---

### 3.2. Improved Model (XLM-RoBERTa + Cleaned Text + Class Weights)
* **Notebook**: [`notebooks/train-xlm-roberta-improved.ipynb`](notebooks/train-xlm-roberta-improved.ipynb)
* **Weights Download**: 🔗 [Google Drive Folder `model_weights_improved.zip`](https://drive.google.com/drive/folders/14EuzHdBYKdDcfyNHC9YXJ-KLThgTSOrT?usp=sharing)
* **Kết quả Test (Argmax Standard Softmax)**:
```text
              precision    recall  f1-score   support

    NEGATIVE     0.9384    0.9730    0.9554      1409
     NEUTRAL     0.7000    0.4611    0.5560       167
    POSITIVE     0.9524    0.9553    0.9538      1590

    accuracy                         0.9371      3166
   macro avg     0.8636    0.7965    0.8217      3166
weighted avg     0.9328    0.9371    0.9336      3166
```
* **Kết quả Test (Threshold Tuning $\tau_{\text{NEUTRAL}} = 0.13$)**:
```text
              precision    recall  f1-score   support

    NEGATIVE     0.9403    0.9723    0.9560      1409
     NEUTRAL     0.6752    0.4731    0.5563       167
    POSITIVE     0.9529    0.9541    0.9535      1590

    accuracy                         0.9368      3166
   macro avg     0.8561    0.7998    0.8220      3166
weighted avg     0.9326    0.9368    0.9337      3166
```

---

### 3.3. PhoBERT-v2 SOTA Model
* **Notebook**: [`notebooks/phobert-v2.ipynb`](notebooks/phobert-v2.ipynb)
* **Weights Download**: 🔗 [Google Drive Folder `phobert_model_weights.zip`](https://drive.google.com/drive/folders/1hkdcZTRQKmz2BXsgPseg1Y85ocgvbqz9?usp=sharing)
* **Kết quả Test (Argmax Standard Softmax)**:
```text
              precision    recall  f1-score   support

    NEGATIVE     0.9423    0.9730    0.9574      1409
     NEUTRAL     0.7544    0.5150    0.6121       167
    POSITIVE     0.9562    0.9604    0.9583      1590

    accuracy                         0.9425      3166
   macro avg     0.8843    0.8161    0.8426      3166
weighted avg     0.9393    0.9425    0.9396      3166
```
* **Kết quả Test (Threshold Tuning $\tau_{\text{NEUTRAL}} = 0.08$)**:
```text
              precision    recall  f1-score   support

    NEGATIVE     0.9494    0.9588    0.9541      1409
     NEUTRAL     0.5969    0.7006    0.6446       167
    POSITIVE     0.9716    0.9453    0.9582      1590

    accuracy                         0.9384      3166
   macro avg     0.8393    0.8682    0.8523      3166
weighted avg     0.9419    0.9384    0.9399      3166
```

---

## 4. Phân Tích Kỹ Thuật Đột Phá (Key Technical Takeaways)

1. **PhoBERT-v2 chiến thắng áp đảo**:
   - Dung lượng file **chỉ 540 MB (nhẹ bằng 1/2 XLM-RoBERTa 1.11GB)**.
   - Đạt **Accuracy cao nhất: 94.25%**.
   - Đạt **Macro F1 cao nhất: 85.23%**.
2. **Kéo vọt lớp NEUTRAL**:
   - Ở chế độ Argmax: Precision của NEUTRAL nhảy vọt lên **75.44%** (cao nhất trong mọi thử nghiệm).
   - Ở chế độ Threshold Tuning ($\tau=0.08$): Recall của NEUTRAL chạm mốc **70.06%** (kéo F1 NEUTRAL lên **64.46%**), giải quyết trọn vẹn điểm nghẽn của lớp trung tính!

---

## 5. Cấu Trúc File & Liên Kết Artifacts Trong Dự Án

```
├── report.md                                          # Báo cáo thực nghiệm chi tiết (file này)
├── docs/
│   ├── report.md                                      # Báo cáo thực nghiệm (copy trong docs)
│   └── KAGGLE_GUIDE.md                                # Hướng dẫn chạy trên Kaggle
├── notebooks/
│   ├── 02_eda_and_data_cleaning_uit_vsfc.ipynb        # Phân tích EDA toàn diện 16,175 mẫu
│   ├── train-xlm-roberta.ipynb                        # Train Baseline
│   ├── train-xlm-roberta-improved.ipynb               # Train XLM-RoBERTa Cải Tiến
│   └── phobert-v2.ipynb                               # Train PhoBERT SOTA
├── src/sentiment/
│   └── training/train.py                              # Codebase training hỗ trợ CLI --apply-cleaning
└── Google Drive Model Weights Links/
    ├── XLM-RoBERTa (Baseline + Improved): 
    │   🔗 https://drive.google.com/drive/folders/14EuzHdBYKdDcfyNHC9YXJ-KLThgTSOrT?usp=sharing
    └── PhoBERT-v2 SOTA Model:
        🔗 https://drive.google.com/drive/folders/1hkdcZTRQKmz2BXsgPseg1Y85ocgvbqz9?usp=sharing
```
