# 🛡️ Network Intrusion Detection ML - Group 14

## 📌 Overview

Dự án xây dựng hệ thống phát hiện xâm nhập mạng (Network Intrusion Detection System - NIDS) sử dụng Machine Learning trên dataset **CIC-IDS2017** của Canadian Institute for Cybersecurity.

---

## 📁 Project Structure

```
Network-Intrusion-Detection-ML/
├── src/
│   ├── 01_eda_preprocessing.py     ← Merge, clean, EDA, lưu pkl
│   └── 02_balancing.py             ← Encode, Scale, SMOTE, UnderSampler
├── dataset/                         ← Đặt 8 file CSV vào đây (xem bên dưới)
│   ├── cleaned_ids_dataset.pkl      ← Output Bước 1
│   ├── X_train_balanced.pkl         ← Output Bước 2
│   ├── X_test.pkl                   ← Output Bước 2
│   ├── y_train_balanced.pkl         ← Output Bước 2
│   ├── y_test.pkl                   ← Output Bước 2
│   ├── scaler.pkl                   ← Output Bước 2
│   ├── label_encoder.pkl            ← Output Bước 2
│   └── feature_names.pkl            ← Output Bước 2
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

## 🔄 Bước 3 — Feature Selection (Đang thực hiện)

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
| Thành viên 3 | Feature Selection (Bước 3) | 🔄 Đang làm |
| Thành viên 4 | Model Training & Evaluation (Bước 4) | ⏳ Chờ |
| Thành viên 5 | Real-time Deployment & README (Bước 5) | ⏳ Chờ |

---

## 🔗 References

- [CIC-IDS2017 Dataset on Kaggle](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset/code)
- [Reference Notebook](https://www.kaggle.com/code/ujjwalks9/intrusion-detection-system)
- [Reference GitHub](https://github.com/marxgoo/Network-intrusion-detection-ml)