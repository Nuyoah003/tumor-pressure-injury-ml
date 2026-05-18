"""
肿瘤患者压力性损伤风险预测 —— 高风险预警模型
================================================
方案: XGBoost + SMOTE + 序数距离代价权重
验证性能: Kappa=0.2477, 高风险Recall=0.4972, AUC(macro)=0.9075

特点:
  - 序数距离代价矩阵 (cost=|i-j|^2) 驱动的样本权重
  - SMOTE过采样平衡训练集
  - 代价敏感XGBoost
  - 验证集概率偏移阈值优化（搜索最优类别边界）
  - 高风险(等级3)召回率最优，适合"不漏检高风险患者"的临床场景

适用场景: 筛查阶段，宁可误报也不漏掉高风险患者

环境要求: Python 3.8+, xgboost, imbalanced-learn, scikit-learn
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, cohen_kappa_score,
    roc_curve, auc as sk_auc, classification_report,
    f1_score, precision_score, recall_score,
)

warnings.filterwarnings("ignore")
matplotlib.rcParams["figure.dpi"] = 150


# ── 中文字体 ──
def _setup_font():
    preferred = ["SimHei", "Microsoft YaHei", "WenQuanYi Zen Hei",
                 "Noto Sans CJK JP", "Noto Serif CJK JP", "PingFang SC"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    return None

_setup_font()


# ════════════════════════════════════════════════════════════
# 配置
# ════════════════════════════════════════════════════════════
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "20260511更正后数据.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR  = os.path.join(BASE_DIR, "saved_models")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET      = "风险等级"
RISK_LABELS = ["无风险(0)", "低风险(1)", "中风险(2)", "高风险(3)"]
K           = 4

BINARY_VARS = [
    "性别", "民族", "是否入住ICU", "放疗史", "化疗史",
    "营养风险", "血栓风险", "高血压", "糖尿病", "骨转移",
    "疼痛", "肿瘤转移", "激素治疗", "免疫抑制剂", "恶性积液",
    "电解质紊乱", "感染", "导管数目",
]
NOMINAL_VARS    = ["肿瘤类型"]
CONTINUOUS_VARS = ["年龄", "住院时长", "白蛋白", "BMI"]


# ════════════════════════════════════════════════════════════
# 序数距离代价矩阵
# ════════════════════════════════════════════════════════════
def build_ordinal_cost_matrix(n_classes=K, power=2):
    """cost[i][j] = |i-j|^power"""
    cost = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            cost[i][j] = abs(i - j) ** power
    return cost


def ordinal_sample_weights(y, cost_matrix):
    """基于代价矩阵的样本权重（纯序数代价，不混逆频率）"""
    y_arr = np.asarray(y, dtype=int).ravel()
    n = len(y_arr)
    n_classes = cost_matrix.shape[0]
    weights = np.ones(n)
    for i in range(n_classes):
        mask = y_arr == i
        avg_cost = (cost_matrix[i].sum() - cost_matrix[i][i]) / (n_classes - 1)
        weights[mask] = avg_cost
    weights /= weights.mean()
    return weights


# ════════════════════════════════════════════════════════════
# 数据准备
# ════════════════════════════════════════════════════════════
def load_and_prepare():
    """加载数据、编码、划分、标准化"""
    df = pd.read_excel(DATA_PATH)
    print(f"[数据加载] {df.shape[0]:,} 行 x {df.shape[1]} 列")

    y = df[TARGET].astype(int)
    X = df.drop(columns=["ID", TARGET, "Braden总分"], errors="ignore")

    for col in NOMINAL_VARS:
        if col not in X.columns:
            continue
        dummies = pd.get_dummies(X[col], prefix=col, drop_first=False, dtype=int)
        dummies = dummies.iloc[:, :-1]
        X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
    X = X.astype(float)
    feat_names = list(X.columns)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for tr, te in sss.split(X, y):
        X_train = X.iloc[tr].reset_index(drop=True)
        X_test = X.iloc[te].reset_index(drop=True)
        y_train = y.iloc[tr].reset_index(drop=True)
        y_test = y.iloc[te].reset_index(drop=True)

    cols = [c for c in CONTINUOUS_VARS if c in X_train.columns]
    scaler = StandardScaler()
    X_train[cols] = scaler.fit_transform(X_train[cols])
    X_test[cols] = scaler.transform(X_test[cols])

    print(f"[数据划分] 训练: {len(X_train):,}  测试: {len(X_test):,}")
    for g in range(K):
        n_tr = (y_train == g).sum()
        n_te = (y_test == g).sum()
        print(f"  {RISK_LABELS[g]}: 训练 {n_tr:,}  测试 {n_te:,}")

    return X_train, X_test, y_train, y_test, scaler, feat_names


# ════════════════════════════════════════════════════════════
# SMOTE 重采样
# ════════════════════════════════════════════════════════════
def apply_smote(X_train, y_train, random_state=42):
    """SMOTE过采样，moderate策略"""
    from imblearn.over_sampling import SMOTE

    counts = y_train.value_counts().sort_index()
    max_count = counts.max()
    target_counts = {}
    for cls in sorted(y_train.unique()):
        c = counts.get(cls, 0)
        target_counts[cls] = max(int(max_count * 0.4), c) if c < max_count else c

    k_neighbors = max(min(5, counts.min() - 1), 1)
    sampler = SMOTE(
        sampling_strategy=target_counts, random_state=random_state,
        k_neighbors=k_neighbors
    )
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    print(f"  [SMOTE] {len(y_train):,} -> {len(y_res):,} 条")
    return X_res, y_res


# ════════════════════════════════════════════════════════════
# 阈值优化
# ════════════════════════════════════════════════════════════
def predict_with_offset(proba, offset=0.0):
    """基于概率偏移量预测类别"""
    if proba.ndim == 1:
        return proba
    adjusted = proba.copy()
    for i in range(K):
        adjusted[:, i] += offset * (K - 1 - 2 * i) / (K - 1)
    return adjusted.argmax(axis=1)


def optimize_threshold(proba, y_true, metric="kappa", n_steps=30):
    """搜索最优概率偏移阈值"""
    y_arr = np.asarray(y_true, dtype=int).ravel()
    best_score = -np.inf
    best_offset = 0.0

    # 粗搜索
    for tf in np.linspace(-0.3, 0.3, n_steps * 3 + 1):
        y_pred = predict_with_offset(proba, offset=tf)
        if metric == "kappa":
            score = cohen_kappa_score(y_arr, y_pred)
        else:
            score = f1_score(y_arr, y_pred, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_offset = tf

    # 精搜索
    tf_lo = max(-0.3, best_offset - 0.05)
    tf_hi = min(0.3, best_offset + 0.05)
    for tf in np.linspace(tf_lo, tf_hi, n_steps):
        y_pred = predict_with_offset(proba, offset=tf)
        if metric == "kappa":
            score = cohen_kappa_score(y_arr, y_pred)
        else:
            score = f1_score(y_arr, y_pred, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_offset = tf

    return best_offset, best_score


# ════════════════════════════════════════════════════════════
# 模型训练
# ════════════════════════════════════════════════════════════
def train_xgb_ordinal_cost(X_tr, y_tr, cost_matrix):
    """XGBoost + SMOTE + 序数距离代价权重"""
    from xgboost import XGBClassifier

    print("  [XGBoost] SMOTE + 序数距离代价权重")
    X_smote, y_smote = apply_smote(X_tr, y_tr)

    # 基于原始分布计算类别权重
    w_map = ordinal_sample_weights(y_tr, cost_matrix)
    unique_classes = sorted(y_tr.unique())
    class_weight_map = {}
    for cls in unique_classes:
        mask = (y_tr == cls)
        class_weight_map[cls] = w_map[mask].mean()

    y_arr = np.asarray(y_smote, dtype=int).ravel()
    sample_weights = np.array([class_weight_map.get(yi, 1.0) for yi in y_arr])
    print(f"  样本权重范围: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")

    clf = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, gamma=0.1,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="multi:softprob", num_class=K,
        eval_metric="mlogloss",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    clf.fit(X_smote.values, y_smote.values, sample_weight=sample_weights)
    print(f"  拟合完成")
    return clf


# ════════════════════════════════════════════════════════════
# 评估
# ════════════════════════════════════════════════════════════
def evaluate_model(y_pred, y_proba, y_test, model_name="高风险预警模型"):
    """完整评估"""
    y_arr = y_test.values if hasattr(y_test, "values") else y_test
    cm = confusion_matrix(y_arr, y_pred)
    kappa = cohen_kappa_score(y_arr, y_pred)

    auc_macro = auc_weighted = None
    per_class_auc = {}
    try:
        if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] == K:
            auc_macro = roc_auc_score(y_arr, y_proba, multi_class="ovr", average="macro")
            auc_weighted = roc_auc_score(y_arr, y_proba, multi_class="ovr", average="weighted")
            y_bin = label_binarize(y_arr, classes=list(range(K)))
            for i in range(K):
                if y_bin[:, i].sum() > 0 and y_bin[:, i].sum() < len(y_bin):
                    fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                    per_class_auc[i] = sk_auc(fpr, tpr)
    except Exception as e:
        print(f"  AUC 计算跳过: {e}")

    macro_f1 = f1_score(y_arr, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_arr, y_pred, average="weighted", zero_division=0)

    per_class_prf = {}
    for i in range(K):
        per_class_prf[i] = {
            "precision": precision_score(y_arr == i, y_pred == i, zero_division=0),
            "recall": recall_score(y_arr == i, y_pred == i, zero_division=0),
            "f1": f1_score(y_arr == i, y_pred == i, zero_division=0),
        }

    recalls = [per_class_prf[i]["recall"] for i in range(K)]
    recalls_pos = [r for r in recalls if r > 0]
    g_mean = np.exp(np.mean(np.log(recalls_pos))) if recalls_pos else 0.0

    # 打印
    print(f"\n{'='*65}")
    print(f"  {model_name} - 评估结果")
    print(f"{'='*65}")
    print(f"\n  混淆矩阵:")
    print(f"  {'':>12} {'预测0':>8} {'预测1':>8} {'预测2':>8} {'预测3':>8}")
    for i in range(K):
        row_pct = cm[i] / cm[i].sum() * 100 if cm[i].sum() > 0 else np.zeros(K)
        row_str = f"  {RISK_LABELS[i]:>12}"
        for j in range(K):
            row_str += f" {cm[i,j]:>5}({row_pct[j]:>4.1f}%)"
        print(row_str)

    print(f"\n  各类别 Precision / Recall / F1:")
    print(f"  {'类别':>12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for i in range(K):
        prf = per_class_prf[i]
        print(f"  {RISK_LABELS[i]:>12} {prf['precision']:>10.4f} {prf['recall']:>10.4f} {prf['f1']:>10.4f}")

    print(f"\n  综合指标:")
    print(f"    Cohen's Kappa:       {kappa:.4f}")
    if auc_macro is not None:
        print(f"    AUC (macro):          {auc_macro:.4f}")
        print(f"    AUC (weighted):       {auc_weighted:.4f}")
    print(f"    Macro F1:             {macro_f1:.4f}")
    print(f"    Weighted F1:          {weighted_f1:.4f}")
    print(f"    G-mean:               {g_mean:.4f}")

    # 高风险预警特有关键指标
    print(f"\n  ★ 高风险预警关键指标:")
    print(f"    高风险Recall (不漏检率): {per_class_prf[3]['recall']:.4f} ({per_class_prf[3]['recall']*100:.1f}%)")
    print(f"    高风险Precision (准确率): {per_class_prf[3]['precision']:.4f}")
    print(f"    无风险Recall (不误报率): {per_class_prf[0]['recall']:.4f} ({per_class_prf[0]['recall']*100:.1f}%)")

    print(f"\n  sklearn classification_report:")
    print(classification_report(y_arr, y_pred, target_names=RISK_LABELS, zero_division=0))

    return {
        "model_name": model_name,
        "kappa": kappa, "auc_macro": auc_macro, "auc_weighted": auc_weighted,
        "macro_f1": macro_f1, "weighted_f1": weighted_f1, "g_mean": g_mean,
        "confusion_matrix": cm, "per_class_prf": per_class_prf,
        "per_class_auc": per_class_auc,
    }


def plot_results(metrics, y_test, y_proba, filename_prefix="high_risk_model"):
    """生成评估图表"""
    y_arr = y_test.values if hasattr(y_test, "values") else y_test
    cm = metrics["confusion_matrix"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 混淆矩阵
    ax = axes[0]
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm[i, j] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0
            annot[i, j] = f"{cm[i, j]}\n({pct:.1f}%)"
    sns.heatmap(cm, annot=annot, fmt="", cmap="OrRd",
                xticklabels=RISK_LABELS, yticklabels=RISK_LABELS,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("预测标签", fontsize=11)
    ax.set_ylabel("真实标签", fontsize=11)
    ax.set_title("混淆矩阵", fontweight="bold")

    # ROC曲线
    ax = axes[1]
    y_bin = label_binarize(y_arr, classes=list(range(K)))
    colors = ["#534AB7", "#1D9E75", "#D85A30", "#E24B4A"]
    for i in range(K):
        if y_bin[:, i].sum() > 0 and y_bin[:, i].sum() < len(y_bin):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            auc_val = sk_auc(fpr, tpr)
            lw = 3 if i == 3 else 1.5
            ax.plot(fpr, tpr, color=colors[i], lw=lw,
                    label=f"{RISK_LABELS[i]} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("假阳性率", fontsize=11)
    ax.set_ylabel("真阳性率", fontsize=11)
    ax.set_title("ROC曲线 (OVR)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"高风险预警模型 (高风险Recall={metrics['per_class_prf'][3]['recall']:.1%}, "
        f"Kappa={metrics['kappa']:.4f})",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, f"{filename_prefix}_evaluation.png")
    plt.savefig(p, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [图表] 评估图 -> {p}")


# ════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════
def train_and_evaluate():
    """完整训练+评估流程"""
    t_start = time.time()
    print("=" * 65)
    print("  肿瘤患者压力性损伤风险预测 - 高风险预警模型")
    print("  方案: XGBoost + SMOTE + 序数距离代价权重 + 阈值优化")
    print("=" * 65)

    # 1. 数据准备
    print("\n[步骤1] 数据准备")
    X_train, X_test, y_train, y_test, scaler, feat_names = load_and_prepare()

    # 2. 从训练集划分验证集（用于阈值优化，与策略6实验一致）
    print("\n[步骤2] 划分验证集")
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=123)
    for tr_idx, val_idx in sss_val.split(X_train, y_train):
        X_tr = X_train.iloc[tr_idx].reset_index(drop=True)
        y_tr = y_train.iloc[tr_idx].reset_index(drop=True)
        X_val = X_train.iloc[val_idx].reset_index(drop=True)
        y_val = y_train.iloc[val_idx].reset_index(drop=True)
    print(f"  训练子集: {len(X_tr):,}  验证子集: {len(X_val):,}  测试集: {len(X_test):,}")

    # 3. 构建代价矩阵
    print("\n[步骤3] 构建序数代价矩阵 (cost=|i-j|^2)")
    cost_matrix = build_ordinal_cost_matrix(n_classes=K, power=2)
    print(cost_matrix)

    # 4. 训练模型（在训练子集上）
    print("\n[步骤4] 训练XGBoost")
    clf = train_xgb_ordinal_cost(X_tr, y_tr, cost_matrix)

    # 5. 阈值优化（在验证集上搜索最优概率偏移）
    print("\n[步骤5] 验证集阈值优化")
    val_proba = clf.predict_proba(X_val.values)
    best_offset, best_val_kappa = optimize_threshold(val_proba, y_val, metric="kappa")
    print(f"  最优阈值偏移: {best_offset:.4f}")
    print(f"  验证集Kappa: {best_val_kappa:.4f}")

    # 6. 测试集评估
    print("\n[步骤6] 测试集评估")
    test_proba = clf.predict_proba(X_test.values)
    y_pred = predict_with_offset(test_proba, offset=best_offset)
    metrics = evaluate_model(y_pred, test_proba, y_test)

    # 7. 生成图表
    print("\n[步骤7] 生成评估图表")
    plot_results(metrics, y_test, test_proba, "high_risk_model")

    # 8. 保存模型
    import joblib
    model_pkg = {
        "model": clf,
        "scaler": scaler,
        "feat_names": feat_names,
        "cost_matrix": cost_matrix,
        "threshold_offset": best_offset,
        "risk_labels": RISK_LABELS,
    }
    model_path = os.path.join(MODEL_DIR, "high_risk_model.pkl")
    joblib.dump(model_pkg, model_path)
    print(f"  [模型] 已保存 -> {model_path}")

    # 汇总
    elapsed = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"  完成! 总耗时 {elapsed:.1f}s")
    print(f"  Kappa={metrics['kappa']:.4f}, AUC(macro)={metrics['auc_macro']:.4f}")
    print(f"  高风险Recall={metrics['per_class_prf'][3]['recall']:.4f}")
    print(f"  阈值偏移={best_offset:.4f}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  模型文件: {model_path}")
    print(f"{'='*65}")


if __name__ == "__main__":
    train_and_evaluate()
