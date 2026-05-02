import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import LinearSVC
from sklearn.naive_bayes     import GaussianNB
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


os.makedirs("dataset", exist_ok=True)
os.makedirs("images",  exist_ok=True)
os.makedirs("models",  exist_ok=True)


# =====================================================
# STEP 1 - LOAD DỮ LIỆU
# =====================================================

print("=" * 65)
print("STEP 1 - LOAD DỮ LIỆU")
print("=" * 65)

required = [
    "dataset/X_train_18.pkl", "dataset/X_test_18.pkl",
    "dataset/y_train_18.pkl", "dataset/y_test_18.pkl",
    "dataset/label_encoder.pkl", "dataset/selected_feature_names.pkl",
]
for f in required:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Không tìm thấy {f}. Hãy chạy 03_feature_selection.py trước.")

X_train    = joblib.load("dataset/X_train_18.pkl")
X_test     = joblib.load("dataset/X_test_18.pkl")
y_train    = joblib.load("dataset/y_train_18.pkl")
y_test     = joblib.load("dataset/y_test_18.pkl")
le         = joblib.load("dataset/label_encoder.pkl")
feat_names = joblib.load("dataset/selected_feature_names.pkl")
y_train    = np.array(y_train)
y_test     = np.array(y_test)

BENIGN_IDX = list(le.classes_).index("BENIGN") if "BENIGN" in le.classes_ else -1

print(f"X_train : {X_train.shape}  |  X_test : {X_test.shape}")
print(f"Classes : {list(le.classes_)}")
print(f"BENIGN index : {BENIGN_IDX}")


# =====================================================
# STEP 2 - SUBSAMPLE CÓ STRATIFY (quan trọng)
# =====================================================

print("\n" + "=" * 65)
print("STEP 2 - CONFIGURE MODELS & SUBSAMPLE")
print("=" * 65)

def stratified_subsample(X, y, n):
    """Lấy n mẫu có stratify → đảm bảo mọi class đều được đại diện."""
    if len(X) <= n:
        return X, y
    _, X_sub, _, y_sub = train_test_split(
        X, y, test_size=n / len(X), stratify=y, random_state=42
    )
    return X_sub, y_sub

n_cls = len(le.classes_)

# ── Kích thước subsample ──────────────────────────────────────────
# LR/NB   : 150K  (~10K/class)  — lbfgs hội tụ nhanh trên dense data
# SVM     : 150K  — LinearSVC fit 1 lần, calibrate trên 30K riêng
# RF      : 500K  — đủ để học, nhanh hơn 2.5M gấp 5×
# KNN tr  :  50K  — brute-force; test cũng subsample để predict nhanh
# KNN te  :  80K
# -----------------------------------------------------------------
X_lr,  y_lr  = stratified_subsample(X_train, y_train, 150_000)
X_rf,  y_rf  = stratified_subsample(X_train, y_train, 500_000)
X_knn, y_knn = stratified_subsample(X_train, y_train,  50_000)
X_test_knn, y_test_knn = stratified_subsample(X_test, y_test, 80_000)

print(f"LR / NB / SVM train : {X_lr.shape[0]:>8,}  (~{150_000//n_cls:,}/class)")
print(f"RF          train   : {X_rf.shape[0]:>8,}  (~{500_000//n_cls:,}/class)")
print(f"KNN         train   : {X_knn.shape[0]:>8,}  KNN test: {X_test_knn.shape[0]:,}")

# SVM: CalibratedClassifierCV(cv=2) trên 50K mẫu
# → LinearSVC fit 2 lần × 25K mỗi fold = nhanh
X_svm, y_svm = stratified_subsample(X_lr, y_lr, 50_000)
print(f"SVM calibration subset : {X_svm.shape[0]:,}")

svm_model = CalibratedClassifierCV(
    LinearSVC(max_iter=3000, random_state=42, dual="auto"),
    cv=2, n_jobs=-1
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=500, tol=1e-3, random_state=42, n_jobs=-1,
        solver="lbfgs",          # lbfgs nhanh hơn saga trên dense data nhỏ
        verbose=0,
    ),
    "SVM (LinearSVC)":  svm_model,
    "Naive Bayes":      GaussianNB(var_smoothing=1e-8),
    "KNN":              KNeighborsClassifier(n_neighbors=5, n_jobs=-1,
                                              algorithm="ball_tree"),
    "Random Forest":    RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1,
        max_depth=20, min_samples_leaf=5,
        verbose=1,               # in tiến trình từng cây
    ),
}

FIT_DATA = {
    "Logistic Regression": (X_lr,   y_lr,   X_test,     y_test),
    "SVM (LinearSVC)":     (X_svm,  y_svm,  X_test,     y_test),
    "Naive Bayes":         (X_lr,   y_lr,   X_test,     y_test),
    "KNN":                 (X_knn,  y_knn,  X_test_knn, y_test_knn),
    "Random Forest":       (X_rf,   y_rf,   X_test,     y_test),
}


# =====================================================
# STEP 3 - TRAIN & EVALUATE
# =====================================================

print("\n" + "=" * 65)
print("STEP 3 - TRAINING & EVALUATION")
print("=" * 65)

results     = {}
all_reports = {}
predictions = {}   # cache (y_eval, y_pred) để tái dùng ở bước vẽ

for name, model in models.items():
    print(f"\n{'─'*55}")
    print(f"[MODEL] {name}")
    print(f"{'─'*55}")

    X_fit, y_fit, X_eval, y_eval = FIT_DATA[name]

    print(f"  Fitting  {X_fit.shape[0]:,} samples...", flush=True)
    t0 = time.time()
    model.fit(X_fit, y_fit)
    train_time = time.time() - t0
    print(f"  Training time : {train_time:.1f}s")

    print(f"  Predicting {X_eval.shape[0]:,} samples...", flush=True)
    t0 = time.time()
    y_pred = model.predict(X_eval)
    pred_time = time.time() - t0
    print(f"  Inference time: {pred_time:.1f}s")

    predictions[name] = (y_eval, y_pred)

    # --- Weighted metrics (tổng quan) ---
    acc      = accuracy_score(y_eval, y_pred)
    prec_w   = precision_score(y_eval, y_pred, average="weighted", zero_division=0)
    rec_w    = recall_score(y_eval, y_pred, average="weighted", zero_division=0)
    f1_w     = f1_score(y_eval, y_pred, average="weighted", zero_division=0)

    # --- Macro metrics (mọi class đều được tính bằng nhau) ---
    prec_m   = precision_score(y_eval, y_pred, average="macro", zero_division=0)
    rec_m    = recall_score(y_eval, y_pred, average="macro", zero_division=0)
    f1_m     = f1_score(y_eval, y_pred, average="macro", zero_division=0)

    # --- Attack Detection Rate: Recall trên CÁC class KHÔNG phải BENIGN ---
    attack_mask = (y_eval != BENIGN_IDX)
    if attack_mask.sum() > 0:
        attack_recall = recall_score(
            y_eval[attack_mask], y_pred[attack_mask],
            average="macro", zero_division=0
        )
    else:
        attack_recall = 0.0

    results[name] = {
        "Accuracy":          round(acc     * 100, 2),
        "Precision (W)":     round(prec_w  * 100, 2),
        "Recall (W)":        round(rec_w   * 100, 2),
        "F1 (Weighted)":     round(f1_w    * 100, 2),
        "F1 (Macro)":        round(f1_m    * 100, 2),
        "Attack Recall (M)": round(attack_recall * 100, 2),
        "Train Time (s)":    round(train_time, 1),
    }

    print(f"\n  Accuracy          : {acc*100:.2f}%")
    print(f"  F1  (Weighted)    : {f1_w*100:.2f}%  ← ảnh hưởng bởi class size")
    print(f"  F1  (Macro)       : {f1_m*100:.2f}%  ← mọi class bằng nhau")
    print(f"  Attack Recall (M) : {attack_recall*100:.2f}%  ← quan trọng nhất cho IDS")

    print(f"\n  Classification Report:\n")
    print(classification_report(y_eval, y_pred, target_names=le.classes_, zero_division=0))

    all_reports[name] = classification_report(
        y_eval, y_pred, target_names=le.classes_, zero_division=0, output_dict=True
    )


# =====================================================
# STEP 4 - BẢNG SO SÁNH
# =====================================================

print("\n" + "=" * 65)
print("STEP 4 - MODEL COMPARISON TABLE")
print("=" * 65)

comparison_df = pd.DataFrame(results).T
comparison_df.index.name = "Model"
print(comparison_df.to_string())

# Best model = F1 Macro (đánh giá công bằng hơn weighted)
best_model_name = comparison_df["F1 (Macro)"].idxmax()
print(f"\nBest Model (by F1 Macro): {best_model_name}")


# =====================================================
# STEP 5 - PER-CLASS ATTACK RECALL (chỉ attack classes)
# =====================================================

print("\n" + "=" * 65)
print("STEP 5 - PER-CLASS ATTACK RECALL")
print("=" * 65)

attack_classes = [c for c in le.classes_ if c != "BENIGN"]
recall_rows    = []

for name in models:
    y_eval, y_pred = predictions[name]
    row = {"Model": name}
    for cls in attack_classes:
        cls_idx  = list(le.classes_).index(cls)
        mask     = (y_eval == cls_idx)
        if mask.sum() > 0:
            row[cls[:15]] = round(recall_score(
                y_eval[mask], y_pred[mask],
                labels=[cls_idx], average="macro", zero_division=0
            ) * 100, 1)
        else:
            row[cls[:15]] = None
    recall_rows.append(row)

recall_df = pd.DataFrame(recall_rows).set_index("Model")
print("\nRecall per attack class (%) — thấp = bỏ sót tấn công:")
print(recall_df.to_string())
print("\n[!] Cột nào < 50% = model bỏ sót hơn nửa tấn công loại đó → nguy hiểm trong IDS")


# =====================================================
# STEP 6 - CONFUSION MATRICES (dùng cache, không predict lại)
# =====================================================

print("\n" + "=" * 65)
print("STEP 6 - PLOT CONFUSION MATRICES")
print("=" * 65)

fig, axes = plt.subplots(2, 3, figsize=(22, 14))
axes = axes.flatten()
short_labels = [c[:12] for c in le.classes_]

for idx, name in enumerate(models):
    y_eval_plot, y_pred_plot = predictions[name]
    cm      = confusion_matrix(y_eval_plot, y_pred_plot, labels=list(range(len(le.classes_))))
    cm_norm = cm.astype(float) / np.where(cm.sum(axis=1, keepdims=True) == 0, 1,
                                           cm.sum(axis=1, keepdims=True))
    sns.heatmap(
        cm_norm, ax=axes[idx],
        cmap="YlOrRd", vmin=0, vmax=1,
        xticklabels=short_labels, yticklabels=short_labels,
        linewidths=0.3, cbar=True,
    )
    f1_val = results[name]["F1 (Macro)"]
    atk    = results[name]["Attack Recall (M)"]
    axes[idx].set_title(f"{name}\nF1-Macro={f1_val:.1f}%  AtkRecall={atk:.1f}%",
                        fontsize=10, fontweight="bold")
    axes[idx].set_xlabel("Predicted", fontsize=8)
    axes[idx].set_ylabel("Actual",    fontsize=8)
    axes[idx].tick_params(axis="x", rotation=45, labelsize=7)
    axes[idx].tick_params(axis="y", rotation=0,  labelsize=7)

axes[5].set_visible(False)
plt.suptitle("Confusion Matrices (Row-normalized = Recall per class)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("images/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: images/confusion_matrices.png")


# =====================================================
# STEP 7 - BIỂU ĐỒ SO SÁNH METRICS
# =====================================================

print("\n" + "=" * 65)
print("STEP 7 - PLOT METRICS COMPARISON")
print("=" * 65)

plot_metrics = ["F1 (Weighted)", "F1 (Macro)", "Attack Recall (M)", "Accuracy"]
x      = np.arange(len(results))
width  = 0.18
colors = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]

fig, ax = plt.subplots(figsize=(15, 6))
for i, metric in enumerate(plot_metrics):
    vals = [results[m][metric] for m in results]
    ax.bar(x + i * width, vals, width, label=metric, color=colors[i], alpha=0.85)

ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels(list(results.keys()), rotation=15, ha="right", fontsize=10)
ax.set_ylabel("Score (%)")
ax.set_title("Model Comparison — Weighted vs Macro vs Attack Recall", fontsize=13, fontweight="bold")
ax.set_ylim(0, 115)
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.axhline(y=80, color="red", linestyle="--", linewidth=1, alpha=0.5, label="80% threshold")

for i, metric in enumerate(plot_metrics):
    for j, m_name in enumerate(results):
        val = results[m_name][metric]
        ax.text(j + i * width, val + 0.5, f"{val:.0f}",
                ha="center", va="bottom", fontsize=6.5, rotation=90)

plt.tight_layout()
plt.savefig("images/model_comparison.png", dpi=150)
plt.show()
print("Saved: images/model_comparison.png")


# =====================================================
# STEP 8 - HEATMAP PER-CLASS ATTACK RECALL
# =====================================================

print("\n" + "=" * 65)
print("STEP 8 - PLOT ATTACK RECALL HEATMAP")
print("=" * 65)

fig, ax = plt.subplots(figsize=(max(10, len(attack_classes) * 0.8), 4))
recall_heat = recall_df.fillna(0).astype(float)
sns.heatmap(
    recall_heat, ax=ax,
    cmap="RdYlGn", vmin=0, vmax=100,
    annot=True, fmt=".0f", linewidths=0.5,
    cbar_kws={"label": "Recall (%)"},
)
ax.set_title("Per-class Attack Recall (%) per Model\n"
             "Xanh = phát hiện tốt  |  Đỏ = bỏ sót nguy hiểm",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Attack Class")
ax.set_ylabel("Model")
ax.tick_params(axis="x", rotation=30, labelsize=8)
plt.tight_layout()
plt.savefig("images/attack_recall_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: images/attack_recall_heatmap.png")


# =====================================================
# STEP 9 - LƯU KẾT QUẢ VÀ MODELS
# =====================================================

print("\n" + "=" * 65)
print("STEP 9 - SAVE RESULTS & MODELS")
print("=" * 65)

comparison_df.to_csv("dataset/model_comparison.csv")
recall_df.to_csv("dataset/attack_recall_per_class.csv")

with open("dataset/classification_reports.json", "w", encoding="utf-8") as f:
    json.dump(all_reports, f, indent=2, ensure_ascii=False)

rf_model = models["Random Forest"]
joblib.dump(rf_model, "models/random_forest.pkl", compress=3)
print("Saved: models/random_forest.pkl")

for name, model in models.items():
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(model, f"models/{safe_name}.pkl", compress=3)
    print(f"Saved: models/{safe_name}.pkl")

print("Saved: dataset/model_comparison.csv")
print("Saved: dataset/attack_recall_per_class.csv")
print("Saved: dataset/classification_reports.json")

print("\n" + "=" * 65)
print("HOÀN THÀNH!")
print("=" * 65)
print(comparison_df[["Accuracy","F1 (Weighted)","F1 (Macro)","Attack Recall (M)"]].to_string())
print(f"\n★ Best Model (F1 Macro): {best_model_name}")
print("=" * 65)