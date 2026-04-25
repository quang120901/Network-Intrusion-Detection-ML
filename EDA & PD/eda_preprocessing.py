import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import VarianceThreshold


# =====================================================
# STEP 1 - LOAD & MERGE 8 CSV FILES
# =====================================================

print("Loading CSV files...")

files = glob.glob("dataset/*.csv")

print("Number of files:", len(files))

df_list = []

for file in files:
    print("Reading:", file)

    temp = pd.read_csv(
        file,
        low_memory=False
        # nếu RAM yếu thêm:
        # nrows=50000
    )

    df_list.append(temp)


df = pd.concat(
    df_list,
    ignore_index=True
)

print("\nMerged shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)



# =====================================================
# STEP 2 - DATA CLEANING
# =====================================================

print("\nCleaning column names...")
df.columns = df.columns.str.strip()


print("Replacing infinite values...")
df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)


# xử lý missing bằng median
numeric_cols = df.select_dtypes(
    include=np.number
).columns

for col in numeric_cols:
    df[col] = df[col].fillna(
        df[col].median()
    )


# vét sạch NaN còn sót
df.fillna(
    0,
    inplace=True
)

print(
    "Null remaining:",
    df.isnull().sum().sum()
)


# xóa duplicate rows
before = df.shape[0]

df.drop_duplicates(
    inplace=True
)

after = df.shape[0]

print(
    "Dropped duplicates:",
    before - after
)



# =====================================================
# STEP 3 - DROP ZERO VARIANCE FEATURES
# =====================================================

print("\nRemoving zero variance features...")

X = df.drop(
    "Label",
    axis=1
)

selector = VarianceThreshold(
    threshold=0
)

X_new = selector.fit_transform(X)

selected_features = X.columns[
    selector.get_support()
]

df = df[
    selected_features.tolist()
    + ['Label']
]

print(
    "Remaining features:",
    len(selected_features)
)



# =====================================================
# STEP 4 - MEMORY OPTIMIZATION
# =====================================================

before_mem = df.memory_usage(
    deep=True
).sum()/1024**2

print(
    "Memory before:",
    before_mem,
    "MB"
)


for c in df.select_dtypes(
    include='int'
).columns:

    df[c] = pd.to_numeric(
        df[c],
        downcast='integer'
    )


for c in df.select_dtypes(
    include='float'
).columns:

    df[c] = pd.to_numeric(
        df[c],
        downcast='float'
    )


after_mem = df.memory_usage(
    deep=True
).sum()/1024**2

print(
    "Memory after:",
    after_mem,
    "MB"
)



# =====================================================
# STEP 5 - ATTACK DISTRIBUTION
# =====================================================

plt.figure(
    figsize=(12,6)
)

df["Label"].value_counts().plot(
    kind="bar"
)

plt.title(
    "Attack Type Distribution"
)

plt.tight_layout()

plt.savefig(
    "attack_distribution.png"
)

plt.show()



# =====================================================
# STEP 6 - CORRELATION HEATMAP
# =====================================================

num_data = df.select_dtypes(
    include=np.number
)

corr = num_data.iloc[:,:15].corr()

plt.figure(
    figsize=(12,8)
)

sns.heatmap(
    corr,
    cmap="coolwarm"
)

plt.title(
    "Correlation Heatmap"
)

plt.tight_layout()

plt.savefig(
    "heatmap.png"
)

plt.show()



# =====================================================
# STEP 7 - SAVE CLEANED DATASET
# =====================================================

df.to_csv(
    "cleaned_ids_dataset.csv",
    index=False
)

print(
    "\nCleaned dataset saved."
)