import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


# =====================================================
# STEP 1 - LOAD CLEANED DATASET TỪ BƯỚC 1
# =====================================================

print("=" * 55)
print("STEP 1 - LOAD CLEANED DATASET")
print("=" * 55)

if not os.path.exists("dataset/cleaned_ids_dataset.pkl"):
    raise FileNotFoundError(
        "Không tìm thấy file cleaned_ids_dataset.pkl.\n"
        "Hãy chạy 01_eda_preprocessing.py trước."
    )

df = joblib.load("dataset/cleaned_ids_dataset.pkl")

print(f"Loaded shape : {df.shape}")
print(f"Columns      : {list(df.columns)}")
print(f"\nLabel distribution (trước khi xử lý):")
print(df["Label"].value_counts())


# =====================================================
# STEP 2 - ENCODE LABEL
# =====================================================

print("\n" + "=" * 55)
print("STEP 2 - ENCODE LABEL")
print("=" * 55)

le = LabelEncoder()
df["Label"] = le.fit_transform(df["Label"])

print("Mapping nhãn (chữ → số):")
for i, cls in enumerate(le.classes_):
    print(f"  {i} ← {cls}")


# =====================================================
# STEP 3 - TÁCH FEATURES VÀ LABEL
# =====================================================

print("\n" + "=" * 55)
print("STEP 3 - TRAIN / TEST SPLIT")
print("=" * 55)

X = df.drop("Label", axis=1)
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y       # giữ tỉ lệ nhãn trong cả train lẫn test
)

print(f"X_train shape : {X_train.shape}")
print(f"X_test shape  : {X_test.shape}")
print(f"y_train distribution:\n{pd.Series(y_train).value_counts()}")


# =====================================================
# STEP 4 - SCALE FEATURES (StandardScaler)
# =====================================================

print("\n" + "=" * 55)
print("STEP 4 - SCALE FEATURES")
print("=" * 55)

scaler = StandardScaler()

# Chỉ fit trên train, transform cả train lẫn test
# QUAN TRỌNG: Không fit trên test → tránh data leakage
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("StandardScaler đã fit trên train set.")
print(f"X_train_scaled shape : {X_train_scaled.shape}")
print(f"X_test_scaled shape  : {X_test_scaled.shape}")


# =====================================================
# STEP 5 - XỬ LÝ CLASS IMBALANCE
# =====================================================

print("\n" + "=" * 55)
print("STEP 5 - HANDLING CLASS IMBALANCE")
print("=" * 55)

# --- Trước khi balance ---
train_counts_before = pd.Series(y_train).value_counts()
print("Phân phối TRƯỚC khi balance:")
print(train_counts_before)

majority_count = train_counts_before.max()
print(f"\nMajority class count : {majority_count:,}")

# --- SMOTE: Over-sample minority lên 10% của majority ---
smote_threshold = int(majority_count * 0.1)
print(f"SMOTE threshold (10%): {smote_threshold:,}")

smote_strategy = {}
for label, count in train_counts_before.items():
    if count < majority_count:
        # Nếu class đang nhỏ hơn threshold → over-sample lên threshold
        # Nếu class đã lớn hơn threshold → giữ nguyên
        target = max(count, smote_threshold)
        smote_strategy[label] = target

print(f"\nSMOTE sampling strategy:\n{smote_strategy}")

smote = SMOTE(
    sampling_strategy=smote_strategy,
    random_state=42,
    k_neighbors=5
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled, y_train
)

print(f"\nPhân phối SAU SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# --- RandomUnderSampler: Under-sample majority class ---
rus = RandomUnderSampler(
    sampling_strategy="majority",   # chỉ giảm class đông nhất
    random_state=42
)

X_train_balanced, y_train_balanced = rus.fit_resample(
    X_train_smote, y_train_smote
)

print(f"\nPhân phối SAU RandomUnderSampler (final):")
final_dist = pd.Series(y_train_balanced).value_counts()
print(final_dist)


# =====================================================
# STEP 6 - VISUALIZE: BEFORE vs AFTER BALANCING
# =====================================================

print("\n" + "=" * 55)
print("STEP 6 - VISUALIZE CLASS DISTRIBUTION")
print("=" * 55)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Trước
before_labels = [le.classes_[i] for i in train_counts_before.index]
axes[0].barh(before_labels, train_counts_before.values, color="salmon")
axes[0].set_title("BEFORE Balancing", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Number of Samples")
axes[0].invert_yaxis()

# Sau
after_dist = pd.Series(y_train_balanced).value_counts()
after_labels = [le.classes_[i] for i in after_dist.index]
axes[1].barh(after_labels, after_dist.values, color="steelblue")
axes[1].set_title("AFTER Balancing (SMOTE + UnderSampler)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Number of Samples")
axes[1].invert_yaxis()

plt.suptitle("Class Distribution: Before vs After Balancing", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("images/class_balance.png", dpi=150)
plt.show()
print("Saved: images/class_balance.png")


# =====================================================
# STEP 7 - LƯU TẤT CẢ OUTPUT CHO BƯỚC 3 & 4
# =====================================================

print("\n" + "=" * 55)
print("STEP 7 - SAVE OUTPUT FILES")
print("=" * 55)

joblib.dump(X_train_balanced, "dataset/X_train_balanced.pkl")
joblib.dump(X_test_scaled,    "dataset/X_test.pkl")
joblib.dump(y_train_balanced, "dataset/y_train_balanced.pkl")
joblib.dump(y_test,           "dataset/y_test.pkl")
joblib.dump(scaler,           "dataset/scaler.pkl")
joblib.dump(le,               "dataset/label_encoder.pkl")

# Lưu thêm danh sách tên cột (người 3 cần để filter features)
joblib.dump(list(X_train.columns), "dataset/feature_names.pkl")

print("Đã lưu:")
print("  dataset/X_train_balanced.pkl  ← features train (đã balance)")
print("  dataset/X_test.pkl            ← features test")
print("  dataset/y_train_balanced.pkl  ← labels train (đã balance)")
print("  dataset/y_test.pkl            ← labels test")
print("  dataset/scaler.pkl            ← StandardScaler (dùng khi deploy)")
print("  dataset/label_encoder.pkl     ← LabelEncoder  (dùng khi deploy)")
print("  dataset/feature_names.pkl     ← tên 18 features (dùng cho bước 3)")

print("\n" + "=" * 55)
print("HOÀN THÀNH CLASS BALANCING!")
print(f"  Train samples (balanced) : {len(y_train_balanced):,}")
print(f"  Test samples             : {len(y_test):,}")
print(f"  Số classes               : {len(le.classes_)}")
print("=" * 55)