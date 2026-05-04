# 🛡️ Network Intrusion Detection ML - Group 14

## 📌 Overview

Dự án xây dựng hệ thống phát hiện xâm nhập mạng (Network Intrusion Detection System - NIDS) sử dụng Machine Learning trên dataset **CIC-IDS2017** của Canadian Institute for Cybersecurity.

---

## 📁 Project Structure

```
Network-Intrusion-Detection-ML/
├── src/
│   ├── 01_eda_preprocessing.py     ← Merge, clean, EDA, lưu pkl
│   ├── 02_balancing.py             ← Encode, Scale, SMOTE, UnderSampler
│   ├── 03_feature_selection.py     ← Giữ 18 feature lõi cho bước 4, 5
│   ├── 04_model_training.py        ← Train và so sánh nhiều model
│   └── 05_realtime_detection.py    ← Mô phỏng phát hiện real-time
├── dataset/                         ← Đặt 8 file CSV vào đây (xem bên dưới)
│   ├── cleaned_ids_dataset.pkl      ← Output Bước 1
│   ├── X_train_balanced.pkl         ← Output Bước 2
│   ├── X_test.pkl                   ← Output Bước 2
│   ├── y_train_balanced.pkl         ← Output Bước 2
│   ├── y_test.pkl                   ← Output Bước 2
│   ├── scaler.pkl                   ← Output Bước 2
│   ├── label_encoder.pkl            ← Output Bước 2
│   ├── feature_names.pkl            ← Output Bước 2
│   ├── X_train_18.pkl               ← Output Bước 3
│   ├── X_test_18.pkl                ← Output Bước 3
│   ├── y_train_18.pkl               ← Output Bước 3
│   ├── y_test_18.pkl                ← Output Bước 3
│   ├── scaler_18.pkl                ← Output Bước 3
│   └── selected_feature_names.pkl   ← Output Bước 3
├── images/
│   ├── attack_distribution.png      ← Biểu đồ phân phối attack types
│   ├── heatmap.png                  ← Correlation heatmap
│   └── class_balance.png            ← Before vs After balancing
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

Dataset sử dụng: **CIC-IDS2017** — Canadian Institute for Cybersecurity

| Thông tin | Chi tiết |
|-----------|----------|
| Số file CSV | 8 files |
| Tổng số features | 79 cột (78 numerical + 1 Label) |
| Loại traffic | BENIGN, DoS, DDoS, PortScan, Web Attack, Infiltration,... |
| Đặc điểm | Highly imbalanced (BENIGN chiếm đa số) |

> ⚠️ **Dataset không được lưu trong repo** vì quá nặng.
> Download tại: [Kaggle - Network Intrusion Dataset](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset/code)
> Sau khi download, đặt tất cả file `.csv` vào thư mục `dataset/`

---

## ⚙️ Installation

**Yêu cầu:** Python 3.8+

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/Network-Intrusion-Detection-ML.git
cd Network-Intrusion-Detection-ML

# 2. Cài thư viện
pip install -r requirements.txt
```

---

## 🚀 Usage

Chạy từng bước theo thứ tự:

```bash
# Bước 1 - EDA & Preprocessing
python src/01_eda_preprocessing.py

# Bước 2 - Class Balancing
python src/02_balancing.py

# Bước 3 - Feature Selection
python src/03_feature_selection.py

# Bước 4 - Model Training
python src/04_model_training.py

# Bước 5 - Real-time Detection
python src/05_realtime_detection.py
```

---

## ✅ Bước 1 — EDA & Preprocessing (Hoàn thành)

File: `src/01_eda_preprocessing.py`

| Bước | Mô tả | Trạng thái |
|------|-------|-----------|
| Load & Merge | Đọc 8 file CSV, gộp thành 1 DataFrame | ✅ |
| Data Cleaning | Xóa whitespace, xử lý inf/NaN bằng median, xóa duplicate | ✅ |
| Zero-Variance | Loại bỏ các cột không có sự biến thiên | ✅ |
| Memory Optimization | Downcast int64→int8/16, float64→float32 | ✅ |
| EDA - Distribution | Vẽ biểu đồ phân phối attack types | ✅ |
| EDA - Heatmap | Vẽ correlation heatmap toàn bộ features | ✅ |
| Save Output | Lưu `cleaned_ids_dataset.pkl` cho bước tiếp theo | ✅ |

**Output:**
- `dataset/cleaned_ids_dataset.pkl` — dùng cho Bước 2
- `images/attack_distribution.png`
- `images/heatmap.png`

---

## ✅ Bước 2 — Class Balancing (Hoàn thành)

File: `src/02_balancing.py`

| Bước | Mô tả | Trạng thái |
|------|-------|-----------|
| Load Data | Đọc `cleaned_ids_dataset.pkl` từ Bước 1 | ✅ |
| Encode Label | Chuyển nhãn chữ → số bằng `LabelEncoder` (15 classes) | ✅ |
| Train/Test Split | Chia 80/20, dùng `stratify` giữ tỉ lệ nhãn | ✅ |
| StandardScaler | Scale features, chỉ fit trên train (tránh data leakage) | ✅ |
| SMOTE | Over-sample minority classes lên 10% của majority | ✅ |
| RandomUnderSampler | Under-sample BENIGN xuống bằng các class khác | ✅ |
| Visualize | Vẽ biểu đồ before vs after balancing | ✅ |
| Save Output | Lưu 7 file pkl cho Bước 3 & 4 | ✅ |

**Kết quả sau balancing:**

| Thông tin | Giá trị |
|-----------|---------|
| Số classes | 15 |
| Mẫu mỗi class (sau balance) | 167,718 |
| Tổng train samples | 2,515,770 |
| Tổng test samples | 504,473 |

**Output:**
- `dataset/X_train_balanced.pkl` — features train đã balance
- `dataset/X_test.pkl` — features test
- `dataset/y_train_balanced.pkl` — labels train đã balance
- `dataset/y_test.pkl` — labels test
- `dataset/scaler.pkl` — StandardScaler (dùng khi deploy)
- `dataset/label_encoder.pkl` — LabelEncoder (dùng khi deploy)
- `dataset/feature_names.pkl` — tên cột (dùng cho Bước 3)
- `images/class_balance.png`

---

## ✅ Bước 3 — Feature Selection (Hoàn thành)

File: `src/03_feature_selection.py`

| Bước | Mô tả | Trạng thái |
|------|-------|-----------|
| Load Output | Đọc dữ liệu đã balance từ Bước 2 | ✅ |
| Core Feature Filter | Giữ lại đúng 18 features lõi theo yêu cầu PDF lab | ✅ |
| Build Scaler | Tạo `scaler_18.pkl` tương thích cho bước real-time | ✅ |
| Save Output | Lưu 6 file pkl cho Bước 4 và 5 | ✅ |

**18 features được giữ lại:**
- `Protocol`
- `Flow Duration`
- `Tot Fwd Pkts`
- `Tot Bwd Pkts`
- `TotLen Fwd Pkts`
- `TotLen Bwd Pkts`
- `Fwd Pkt Len Mean`
- `Bwd Pkt Len Mean`
- `Flow Byts/s`
- `Flow Pkts/s`
- `Pkt Len Mean`
- `Pkt Len Std`
- `SYN Flag Cnt`
- `ACK Flag Cnt`
- `FIN Flag Cnt`
- `RST Flag Cnt`
- `PSH Flag Cnt`
- `URG Flag Cnt`

**Output:**
- `dataset/X_train_18.pkl`
- `dataset/X_test_18.pkl`
- `dataset/y_train_18.pkl`
- `dataset/y_test_18.pkl`
- `dataset/scaler_18.pkl`
- `dataset/selected_feature_names.pkl`

---

## 📦 Model Comparison (Cập nhật sau)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | - | - | - | - |
| SVM | - | - | - | - |
| Naive Bayes | - | - | - | - |
| KNN | - | - | - | - |
| **Random Forest** | - | - | - | - |

> Bảng sẽ được cập nhật sau khi hoàn thành Bước 4 - Model Training.

---

## 👥 Nhóm 14

| Thành viên | Phụ trách | Trạng thái |
|------------|-----------|-----------|
| Quỳnh | EDA & Preprocessing (Bước 1) | ✅ Hoàn thành |
| Quang | Class Balancing - SMOTE (Bước 2) | ✅ Hoàn thành |
| Minh Tâm | Feature Selection (Bước 3) | ✅ Hoàn thành |
| Hoài Tâm | Model Training & Evaluation (Bước 4) | ✅ Hoàn thành |
| Quân | Real-time Deployment & README (Bước 5) | ✅ Hoàn thành |

---

## 🔗 References

- [CIC-IDS2017 Dataset on Kaggle](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset/code)
- [Reference Notebook](https://www.kaggle.com/code/ujjwalks9/intrusion-detection-system)
- [Reference GitHub](https://github.com/marxgoo/Network-intrusion-detection-ml)
