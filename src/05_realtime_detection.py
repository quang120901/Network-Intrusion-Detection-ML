"""
Real-time Network Intrusion Detection System
Sử dụng mô hình Random Forest đã được train để phân loại
network traffic và tạo cảnh báo kiểu Suricata.
"""

import os
import time
import random
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime


# =====================================================
# CẤU HÌNH LOGGING — GHI FILE alerts.log
# =====================================================

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("logs/alerts.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),          # in ra console
    ],
)
logger = logging.getLogger("NIDS")


# =====================================================
# STEP 1 - LOAD MÔ HÌNH VÀ CÁC ARTIFACT
# =====================================================

print("=" * 65)
print("STEP 1 - LOAD MODEL & ARTIFACTS")
print("=" * 65)

required = [
    "models/random_forest.pkl",
    "dataset/scaler_18.pkl",
    "dataset/label_encoder.pkl",
    "dataset/selected_feature_names.pkl",
]
for f in required:
    if not os.path.exists(f):
        raise FileNotFoundError(
            f"Không tìm thấy {f}.\n"
            "Hãy chạy 03_feature_selection.py và 04_model_training.py trước."
        )

rf_model      = joblib.load("models/random_forest.pkl")
scaler_18     = joblib.load("dataset/scaler_18.pkl")
le            = joblib.load("dataset/label_encoder.pkl")
feature_names = joblib.load("dataset/selected_feature_names.pkl")

print(f"Model         : Random Forest ({rf_model.n_estimators} estimators)")
print(f"Features      : {len(feature_names)} — {feature_names}")
print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")

BENIGN_LABEL = "BENIGN"
BENIGN_IDX   = list(le.classes_).index(BENIGN_LABEL) if BENIGN_LABEL in le.classes_ else -1

# Priority mapping dựa trên loại tấn công
PRIORITY_MAP = {
    "DoS":         1,
    "DDoS":        1,
    "PortScan":    2,
    "Infiltration":1,
    "Web Attack":  2,
    "Heartbleed":  1,
    "Bot":         1,
    "FTP-Patator": 2,
    "SSH-Patator": 2,
}


# =====================================================
# STEP 2 - HÀM XỬ LÝ VÀ PHÂN LOẠI MỘT FLOW
# =====================================================

def analyze_flow(raw_features: dict) -> dict:
    """
    Nhận một network flow dưới dạng dict {feature_name: value},
    scale và phân loại bằng Random Forest.

    Trả về dict gồm:
      - prediction     : tên class (str)
      - confidence     : xác suất cao nhất (float)
      - is_attack      : bool
      - probabilities  : dict {class_name: prob}
    """
    # Sắp xếp đúng thứ tự features
    vector = np.array(
        [raw_features.get(f, 0.0) for f in feature_names],
        dtype=np.float32
    ).reshape(1, -1)

    # Scale
    vector_scaled = scaler_18.transform(vector)

    # Predict
    pred_idx  = rf_model.predict(vector_scaled)[0]
    pred_name = le.inverse_transform([pred_idx])[0]

    # Xác suất
    if hasattr(rf_model, "predict_proba"):
        probs_arr    = rf_model.predict_proba(vector_scaled)[0]
        confidence   = float(probs_arr.max())
        probabilities = {le.classes_[i]: round(float(p), 4)
                         for i, p in enumerate(probs_arr)}
    else:
        confidence    = 1.0
        probabilities = {pred_name: 1.0}

    return {
        "prediction":    pred_name,
        "confidence":    confidence,
        "is_attack":     pred_name != BENIGN_LABEL,
        "probabilities": probabilities,
    }


# =====================================================
# STEP 3 - HÀM TẠO CẢNH BÁO KIỂU SURICATA
# =====================================================

def generate_alert(result: dict, flow_meta: dict | None = None) -> str:
    """
    Tạo chuỗi cảnh báo theo định dạng Suricata EVE-style.
    flow_meta (tuỳ chọn): dict chứa src_ip, dst_ip, dst_port, proto.
    """
    if not result["is_attack"]:
        return ""

    attack_type = result["prediction"]
    confidence  = result["confidence"] * 100

    # Tìm priority
    priority = 3  # mặc định
    for key, prio in PRIORITY_MAP.items():
        if key.lower() in attack_type.lower():
            priority = prio
            break

    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    # Thông tin mạng (mô phỏng nếu không có)
    if flow_meta:
        src_ip   = flow_meta.get("src_ip",   _random_ip())
        dst_ip   = flow_meta.get("dst_ip",   _random_ip())
        dst_port = flow_meta.get("dst_port", random.randint(1, 65535))
        proto    = flow_meta.get("proto",    "TCP")
    else:
        src_ip   = _random_ip()
        dst_ip   = _random_ip()
        dst_port = random.choice([80, 443, 22, 21, 8080, 3306, 53])
        proto    = random.choice(["TCP", "UDP"])

    # Format Suricata-style
    alert_str = (
        f"[{ts}] "
        f"[**] [1:2{random.randint(100000,999999)}:1] "
        f"NIDS ALERT — Suspicious traffic detected: {attack_type} [**] "
        f"[Classification: Network Attack] "
        f"[Priority: {priority}] "
        f"[Confidence: {confidence:.1f}%] "
        f"{{{proto}}} {src_ip} -> {dst_ip}:{dst_port}"
    )
    return alert_str


def _random_ip() -> str:
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


# =====================================================
# STEP 4 - SIMULINK REAL-TIME (dùng mẫu từ test set)
# =====================================================

print("\n" + "=" * 65)
print("STEP 4 - REAL-TIME SIMULATION")
print("=" * 65)

if not os.path.exists("dataset/X_test_18.pkl") or not os.path.exists("dataset/y_test_18.pkl"):
    raise FileNotFoundError("Không tìm thấy X_test_18.pkl / y_test_18.pkl.")

X_test = joblib.load("dataset/X_test_18.pkl")
y_test = joblib.load("dataset/y_test_18.pkl")
y_test_arr = np.array(y_test)

print(f"Test set size: {len(X_test):,} flows")
print(f"Bắt đầu mô phỏng real-time phân loại...\n")

# Chọn 30 flows ngẫu nhiên (cố gắng lấy đủ cả benign lẫn attack)
rng = np.random.default_rng(2024)
attack_idx = np.where(y_test_arr != BENIGN_IDX)[0]
benign_idx = np.where(y_test_arr == BENIGN_IDX)[0]

n_attack = min(20, len(attack_idx))
n_benign = min(10, len(benign_idx))

chosen_attack = rng.choice(attack_idx, n_attack, replace=False)
chosen_benign = rng.choice(benign_idx, n_benign, replace=False)
chosen_indices = np.concatenate([chosen_attack, chosen_benign])
rng.shuffle(chosen_indices)

# Counters
total_flows = 0
total_alerts = 0
true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0

logger.info("=" * 65)
logger.info("NIDS REAL-TIME MONITORING LOG")
logger.info(f"Session start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("=" * 65)
logger.info("")

for i, idx in enumerate(chosen_indices):
    # Lấy raw feature vector
    raw_vec = X_test[idx]  # đã scaled → giả lập raw bằng cách inverse_transform
    raw_unscaled = scaler_18.inverse_transform(raw_vec.reshape(1, -1))[0]

    raw_features = {name: float(val) for name, val in zip(feature_names, raw_unscaled)}

    # Phân loại
    result = analyze_flow(raw_features)

    # Nhãn thực tế
    true_label = le.inverse_transform([y_test_arr[idx]])[0]
    total_flows += 1

    # Tính confusion
    if result["is_attack"] and true_label != BENIGN_LABEL:
        true_positive += 1
    elif result["is_attack"] and true_label == BENIGN_LABEL:
        false_positive += 1
    elif not result["is_attack"] and true_label == BENIGN_LABEL:
        true_negative += 1
    else:
        false_negative += 1

    # Log tất cả flows
    status = "ALERT" if result["is_attack"] else "OK   "
    logger.info(
        f"[{status}] Flow #{i+1:03d} | "
        f"Predicted: {result['prediction']:<30} | "
        f"Actual: {true_label:<30} | "
        f"Confidence: {result['confidence']*100:.1f}%"
    )

    # Tạo alert nếu là attack
    if result["is_attack"]:
        total_alerts += 1
        alert_str = generate_alert(result)
        logger.info(f"        {alert_str}")
        logger.info("")

    # Mô phỏng delay xử lý (0 delay thật sự, chỉ minh họa)
    time.sleep(0)   # set > 0 nếu muốn thấy real-time effect

# =====================================================
# STEP 5 - BÁO CÁO KẾT QUẢ
# =====================================================

print("\n" + "=" * 65)
print("STEP 5 - DETECTION SUMMARY")
print("=" * 65)

precision_rt = true_positive / (true_positive + false_positive + 1e-9)
recall_rt    = true_positive / (true_positive + false_negative + 1e-9)
f1_rt        = 2 * precision_rt * recall_rt / (precision_rt + recall_rt + 1e-9)

summary = f"""
╔══════════════════════════════════════════════╗
║        REAL-TIME IDS DETECTION SUMMARY       ║
╠══════════════════════════════════════════════╣
║  Total flows analyzed  : {total_flows:<20}║
║  Alerts generated      : {total_alerts:<20}║
║──────────────────────────────────────────────║
║  True  Positives (TP)  : {true_positive:<20}║
║  False Positives (FP)  : {false_positive:<20}║
║  True  Negatives (TN)  : {true_negative:<20}║
║  False Negatives (FN)  : {false_negative:<20}║
║──────────────────────────────────────────────║
║  Precision             : {precision_rt*100:<19.2f}%║
║  Recall                : {recall_rt*100:<19.2f}%║
║  F1-Score              : {f1_rt*100:<19.2f}%║
╚══════════════════════════════════════════════╝
"""

print(summary)
logger.info("")
logger.info("=" * 65)
logger.info(summary)
logger.info(f"Session end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("=" * 65)

print("Saved: logs/alerts.log")


# =====================================================
# STEP 6 - VÍ DỤ THỦ CÔNG VỚI MỘT FLOW
# =====================================================

print("\n" + "=" * 65)
print("STEP 6 - MANUAL SINGLE-FLOW EXAMPLE")
print("=" * 65)

# Tạo flow giả định giống DDoS: nhiều gói nhỏ, tốc độ cao
example_flow = {
    "Protocol":          6,          # TCP
    "Flow Duration":     100,
    "Tot Fwd Pkts":      500,
    "Tot Bwd Pkts":      10,
    "TotLen Fwd Pkts":   5000,
    "TotLen Bwd Pkts":   200,
    "Fwd Pkt Len Mean":  10.0,
    "Bwd Pkt Len Mean":  20.0,
    "Flow Byts/s":       1_000_000,
    "Flow Pkts/s":       5_000,
    "Pkt Len Mean":      10.5,
    "Pkt Len Std":       2.1,
    "SYN Flag Cnt":      100,
    "ACK Flag Cnt":      50,
    "FIN Flag Cnt":      0,
    "RST Flag Cnt":      5,
    "PSH Flag Cnt":      0,
    "URG Flag Cnt":      0,
}

# Điền 0 cho features còn thiếu
for f in feature_names:
    example_flow.setdefault(f, 0.0)

result = analyze_flow(example_flow)
print(f"\nInput flow features:")
for k, v in example_flow.items():
    if k in feature_names:
        print(f"  {k:<25}: {v}")

print(f"\nPrediction : {result['prediction']}")
print(f"Is Attack  : {result['is_attack']}")
print(f"Confidence : {result['confidence']*100:.2f}%")
print(f"\nTop-3 class probabilities:")
sorted_probs = sorted(result["probabilities"].items(), key=lambda x: -x[1])
for cls, prob in sorted_probs[:3]:
    print(f"  {cls:<35}: {prob*100:.2f}%")

if result["is_attack"]:
    alert = generate_alert(result, flow_meta={
        "src_ip": "192.168.1.105", "dst_ip": "10.0.0.1",
        "dst_port": 80, "proto": "TCP"
    })
    print(f"\n{alert}")
else:
    print("\n[OK] Traffic classified as BENIGN — no alert generated.")

print("\n" + "=" * 65)
print("HOÀN THÀNH REAL-TIME DETECTION!")
print("=" * 65)