import os
import glob
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import VarianceThreshold


# =====================================================
# TẠO THƯ MỤC OUTPUT NẾU CHƯA CÓ
# =====================================================

os.makedirs("dataset", exist_ok=True)
os.makedirs("images", exist_ok=True)


# =====================================================
# STEP 1 - LOAD & MERGE 8 CSV FILES
# =====================================================

print("=" * 55)
print("STEP 1 - LOAD & MERGE CSV FILES")
print("=" * 55)

files = glob.glob("dataset/*.csv")

print(f"Number of files found: {len(files)}")

if len(files) == 0:
    raise FileNotFoundError(
        "Không tìm thấy file CSV nào trong thư mục dataset/. "
        "Hãy chắc chắn đã đặt các file CSV vào đúng thư mục."
    )

df_list = []

for file in files:
    print(f"  Reading: {file}")
    temp = pd.read_csv(file, low_memory=False)
    print(f"    → Shape: {temp.shape}")
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)

print(f"\nMerged shape: {df.shape}")
print(f"Columns ({len(df.columns)}):\n{list(df.columns)}")


# =====================================================
# STEP 2 - DATA CLEANING
# =====================================================

print("\n" + "=" * 55)
print("STEP 2 - DATA CLEANING")
print("=" * 55)

# Xóa khoảng trắng tên cột
print("Cleaning column names...")
df.columns = df.columns.str.strip()

# Thay inf bằng NaN
print("Replacing infinite values with NaN...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Điền NaN bằng median của từng cột số
print("Filling NaN with column median...")
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Vét sạch NaN còn sót (cột non-numeric)
df.fillna(0, inplace=True)

print(f"Null values remaining: {df.isnull().sum().sum()}")

# Xóa duplicate rows
before = df.shape[0]
df.drop_duplicates(inplace=True)
after = df.shape[0]
print(f"Dropped duplicate rows: {before - after}")
print(f"Shape after cleaning: {df.shape}")


# =====================================================
# STEP 3 - DROP ZERO VARIANCE FEATURES
# =====================================================

print("\n" + "=" * 55)
print("STEP 3 - DROP ZERO VARIANCE FEATURES")
print("=" * 55)

X = df.drop("Label", axis=1)

selector = VarianceThreshold(threshold=0)
selector.fit_transform(X)

selected_features = X.columns[selector.get_support()]

removed = len(X.columns) - len(selected_features)
print(f"Removed zero-variance features: {removed}")
print(f"Remaining features: {len(selected_features)}")

df = df[selected_features.tolist() + ['Label']]


# =====================================================
# STEP 4 - MEMORY OPTIMIZATION
# =====================================================

print("\n" + "=" * 55)
print("STEP 4 - MEMORY OPTIMIZATION")
print("=" * 55)

before_mem = df.memory_usage(deep=True).sum() / 1024**2
print(f"Memory before: {before_mem:.2f} MB")

# Downcast integer columns
for c in df.select_dtypes(include='int').columns:
    df[c] = pd.to_numeric(df[c], downcast='integer')

# Downcast float columns
for c in df.select_dtypes(include='float').columns:
    df[c] = pd.to_numeric(df[c], downcast='float')

after_mem = df.memory_usage(deep=True).sum() / 1024**2
print(f"Memory after:  {after_mem:.2f} MB")
print(f"Reduced by:    {before_mem - after_mem:.2f} MB ({(1 - after_mem/before_mem)*100:.1f}%)")


# =====================================================
# STEP 5 - EDA: ATTACK DISTRIBUTION
# =====================================================

print("\n" + "=" * 55)
print("STEP 5 - EDA: ATTACK TYPE DISTRIBUTION")
print("=" * 55)

label_counts = df["Label"].value_counts()
print("Label distribution:")
print(label_counts)

plt.figure(figsize=(14, 6))
ax = sns.barplot(
    x=label_counts.values,
    y=label_counts.index,
    palette="viridis"
)

# Ghi số lượng lên mỗi bar
for i, v in enumerate(label_counts.values):
    ax.text(v + 500, i, f"{v:,}", va='center', fontsize=9)

plt.title("Attack Type Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Number of Samples")
plt.ylabel("Traffic Type")
plt.tight_layout()
plt.savefig("images/attack_distribution.png", dpi=150)
plt.show()
print("Saved: images/attack_distribution.png")


# =====================================================
# STEP 6 - EDA: CORRELATION HEATMAP
# =====================================================

print("\n" + "=" * 55)
print("STEP 6 - EDA: CORRELATION HEATMAP")
print("=" * 55)

num_data = df.select_dtypes(include=np.number)

# Lấy toàn bộ features (không cắt 15 cột như trước)
corr = num_data.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    xticklabels=False,  # ẩn label trục x vì quá nhiều cột
    yticklabels=False   # ẩn label trục y vì quá nhiều cột
)
plt.title("Correlation Heatmap of All Numerical Features", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("images/heatmap.png", dpi=150)
plt.show()
print("Saved: images/heatmap.png")


# =====================================================
# STEP 7 - SAVE CLEANED DATASET
# =====================================================

print("\n" + "=" * 55)
print("STEP 7 - SAVE CLEANED DATASET")
print("=" * 55)

# Lưu dạng pkl (nhanh hơn CSV, dùng cho các bước tiếp theo)
joblib.dump(df, "dataset/cleaned_ids_dataset.pkl")
print("Saved: dataset/cleaned_ids_dataset.pkl  ← dùng cho bước 2")

# Lưu thêm CSV để dễ kiểm tra bằng mắt nếu cần
df.to_csv("dataset/cleaned_ids_dataset.csv", index=False)
print("Saved: dataset/cleaned_ids_dataset.csv  ← xem thủ công")

print("\n" + "=" * 55)
print("HOÀN THÀNH PREPROCESSING!")
print(f"  Final shape       : {df.shape}")
print(f"  Total labels      : {df['Label'].nunique()} loại")
print(f"  Memory usage      : {after_mem:.2f} MB")
print("=" * 55)