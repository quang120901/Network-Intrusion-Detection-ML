import os
import joblib
import numpy as np

from sklearn.preprocessing import StandardScaler


os.makedirs("dataset", exist_ok=True)


# =====================================================
# STEP 1 - LOAD OUTPUTS TU BUOC 2
# =====================================================

print("=" * 60)
print("STEP 1 - LOAD BALANCED DATA")
print("=" * 60)

required = [
    "dataset/X_train_balanced.pkl",
    "dataset/X_test.pkl",
    "dataset/y_train_balanced.pkl",
    "dataset/y_test.pkl",
    "dataset/scaler.pkl",
    "dataset/feature_names.pkl",
]
for file_path in required:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Khong tim thay {file_path}. Hay chay 02_balancing.py truoc."
        )

X_train_balanced = np.asarray(joblib.load("dataset/X_train_balanced.pkl"))
X_test = np.asarray(joblib.load("dataset/X_test.pkl"))
y_train_balanced = np.asarray(joblib.load("dataset/y_train_balanced.pkl"))
y_test = np.asarray(joblib.load("dataset/y_test.pkl"))
scaler_full = joblib.load("dataset/scaler.pkl")
feature_names = list(joblib.load("dataset/feature_names.pkl"))

print(f"X_train_balanced : {X_train_balanced.shape}")
print(f"X_test           : {X_test.shape}")
print(f"So feature hien co: {len(feature_names)}")


# =====================================================
# STEP 2 - CHON 18 CORE FEATURES THEO PDF
# =====================================================

print("\n" + "=" * 60)
print("STEP 2 - SELECT 18 CORE FEATURES")
print("=" * 60)

pdf_selected_features = [
    "Protocol",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Fwd Pkt Len Mean",
    "Bwd Pkt Len Mean",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Pkt Len Mean",
    "Pkt Len Std",
    "SYN Flag Cnt",
    "ACK Flag Cnt",
    "FIN Flag Cnt",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "URG Flag Cnt",
]

feature_aliases = {
    "Protocol": ["Protocol"],
    "Flow Duration": ["Flow Duration"],
    "Tot Fwd Pkts": ["Tot Fwd Pkts", "Total Fwd Packets"],
    "Tot Bwd Pkts": ["Tot Bwd Pkts", "Total Backward Packets"],
    "TotLen Fwd Pkts": ["TotLen Fwd Pkts", "Total Length of Fwd Packets"],
    "TotLen Bwd Pkts": ["TotLen Bwd Pkts", "Total Length of Bwd Packets"],
    "Fwd Pkt Len Mean": ["Fwd Pkt Len Mean", "Fwd Packet Length Mean"],
    "Bwd Pkt Len Mean": ["Bwd Pkt Len Mean", "Bwd Packet Length Mean"],
    "Flow Byts/s": ["Flow Byts/s", "Flow Bytes/s"],
    "Flow Pkts/s": ["Flow Pkts/s", "Flow Packets/s"],
    "Pkt Len Mean": ["Pkt Len Mean", "Packet Length Mean"],
    "Pkt Len Std": ["Pkt Len Std", "Packet Length Std"],
    "SYN Flag Cnt": ["SYN Flag Cnt", "SYN Flag Count"],
    "ACK Flag Cnt": ["ACK Flag Cnt", "ACK Flag Count"],
    "FIN Flag Cnt": ["FIN Flag Cnt", "FIN Flag Count"],
    "RST Flag Cnt": ["RST Flag Cnt", "RST Flag Count"],
    "PSH Flag Cnt": ["PSH Flag Cnt", "PSH Flag Count"],
    "URG Flag Cnt": ["URG Flag Cnt", "URG Flag Count"],
}

resolved_features = []
selected_indices = []
optional_missing = []
required_missing = []

for canonical_name in pdf_selected_features:
    actual_name = next(
        (candidate for candidate in feature_aliases[canonical_name] if candidate in feature_names),
        None,
    )

    if actual_name is None:
        if canonical_name == "Protocol":
            optional_missing.append(canonical_name)
            continue
        required_missing.append(canonical_name)
        continue

    resolved_features.append((canonical_name, actual_name))
    selected_indices.append(feature_names.index(actual_name))

if required_missing:
    raise ValueError(
        "Khong tim thay mot so feature can thiet trong dataset: "
        + ", ".join(required_missing)
    )

selected_feature_names = [canonical for canonical, _ in resolved_features]

X_train_18 = X_train_balanced[:, selected_indices]
X_test_18 = X_test[:, selected_indices]

if optional_missing:
    print(
        "[WARNING] Khong tim thay feature tuy chon trong dataset hien tai: "
        + ", ".join(optional_missing)
    )
    print("          Script se tiep tuc voi cac feature con lai.")

print("Selected feature mapping:")
for idx, (canonical_name, actual_name) in enumerate(resolved_features, start=1):
    print(f"  {idx:02d}. {canonical_name}  ->  {actual_name}")

print(f"\nX_train_18 shape : {X_train_18.shape}")
print(f"X_test_18 shape  : {X_test_18.shape}")


# =====================================================
# STEP 3 - TAO SCALER RIENG CHO 18 FEATURES
# =====================================================

print("\n" + "=" * 60)
print("STEP 3 - BUILD SCALER FOR 18 FEATURES")
print("=" * 60)

scaler_18 = StandardScaler(
    copy=getattr(scaler_full, "copy", True),
    with_mean=getattr(scaler_full, "with_mean", True),
    with_std=getattr(scaler_full, "with_std", True),
)

if hasattr(scaler_full, "mean_"):
    scaler_18.mean_ = np.asarray(scaler_full.mean_)[selected_indices].copy()

if hasattr(scaler_full, "scale_"):
    scaler_18.scale_ = np.asarray(scaler_full.scale_)[selected_indices].copy()

if hasattr(scaler_full, "var_"):
    scaler_18.var_ = np.asarray(scaler_full.var_)[selected_indices].copy()

if hasattr(scaler_full, "n_samples_seen_"):
    scaler_18.n_samples_seen_ = scaler_full.n_samples_seen_

scaler_18.n_features_in_ = len(selected_feature_names)

print("Da tao scaler_18 tu scaler goc cua buoc 2.")
print(f"So feature trong scaler_18: {scaler_18.n_features_in_}")


# =====================================================
# STEP 4 - LUU OUTPUT CHO BUOC 4 & 5
# =====================================================

print("\n" + "=" * 60)
print("STEP 4 - SAVE OUTPUT FILES")
print("=" * 60)

joblib.dump(X_train_18, "dataset/X_train_18.pkl")
joblib.dump(X_test_18, "dataset/X_test_18.pkl")
joblib.dump(y_train_balanced, "dataset/y_train_18.pkl")
joblib.dump(y_test, "dataset/y_test_18.pkl")
joblib.dump(selected_feature_names, "dataset/selected_feature_names.pkl")
joblib.dump(scaler_18, "dataset/scaler_18.pkl")

print("Da luu:")
print("  dataset/X_train_18.pkl")
print("  dataset/X_test_18.pkl")
print("  dataset/y_train_18.pkl")
print("  dataset/y_test_18.pkl")
print("  dataset/selected_feature_names.pkl")
print("  dataset/scaler_18.pkl")

print("\n" + "=" * 60)
print("HOAN THANH FEATURE SELECTION!")
print("=" * 60)
