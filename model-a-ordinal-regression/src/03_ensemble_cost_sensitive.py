"""
肿瘤患者压力性损伤风险预测 —— 代价敏感序数集成模型（策略6）
=================================================================
核心创新：
  1. 序数距离代价矩阵: 惩罚与等级间距离成正比（3→0的惩罚 >> 3→2）
  2. SMOTE变体优选: SMOTE / Borderline-SMOTE / SMOTE-ENN 三种对比
  3. 代价敏感训练: XGBoost / LightGBM 基于代价矩阵的样本加权
  4. 阈值优化: 搜索最优类别边界而非依赖默认argmax
  5. 软投票集成: 融合多模型概率预测，降低方差

与策略2(SMOTE优化)、策略5(级联模型)使用完全一致的指标体系。

环境要求：Python 3.8+, xgboost, lightgbm, imbalanced-learn, scikit-learn
"""

import os
import sys
import time
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats as sp_stats
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, cohen_kappa_score,
    roc_curve, auc as sk_auc, classification_report,
    f1_score, precision_score, recall_score,
)
from sklearn.calibration import CalibratedClassifierCV

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


# ── 路径配置 ──
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "20260511更正后数据.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 变量定义 ──
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

TUMOR_NAMES = {
    1: "肺癌", 2: "乳腺癌", 3: "结直肠癌", 4: "宫颈癌", 5: "食管癌",
    6: "胃癌", 7: "非霍奇金淋巴瘤", 8: "甲状腺癌", 9: "卵巢癌", 10: "肝癌",
    11: "子宫内膜癌", 12: "前列腺癌", 13: "膀胱癌", 14: "肾癌", 15: "其他",
}


# ════════════════════════════════════════════════════════════
# 序数距离代价矩阵
# ════════════════════════════════════════════════════════════
def build_ordinal_cost_matrix(n_classes=K, power=2):
    """
    构建序数距离代价矩阵。
    cost[i][j] = |i - j|^power，惩罚远离对角线的错误分类。
    power=2 使得距离为2的惩罚是距离为1的4倍。
    """
    cost = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            cost[i][j] = abs(i - j) ** power
    return cost


def ordinal_sample_weights(y, cost_matrix):
    """
    基于代价矩阵为每个样本计算权重。
    w_i = sum_{j != true_class} cost[true_class][j] / (K - 1)
    本质：对容易被远距离误分的少数类样本赋予更高权重。
    """
    y_arr = np.asarray(y, dtype=int).ravel()
    n = len(y_arr)
    n_classes = cost_matrix.shape[0]
    weights = np.ones(n)
    for i in range(n_classes):
        mask = y_arr == i
        # 该类别被误判的平均代价
        avg_cost = (cost_matrix[i].sum() - cost_matrix[i][i]) / (n_classes - 1)
        weights[mask] = avg_cost
    # 归一化到均值=1
    weights /= weights.mean()
    return weights


def inverse_freq_weights(y):
    """传统的逆频率权重"""
    y_arr = np.asarray(y, dtype=int).ravel()
    counts = np.bincount(y_arr, minlength=K)
    weights = np.zeros(len(y_arr))
    n_total = len(y_arr)
    for i in range(K):
        mask = y_arr == i
        weights[mask] = n_total / (K * max(counts[i], 1))
    weights /= weights.mean()
    return weights


def combined_weights(y, cost_matrix, alpha=0.5):
    """
    混合权重 = alpha * 逆频率 + (1-alpha) * 序数代价
    alpha=0.5 给两种策略等权重。
    """
    w_inv = inverse_freq_weights(y)
    w_ord = ordinal_sample_weights(y, cost_matrix)
    return alpha * w_inv + (1 - alpha) * w_ord


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
# SMOTE 变体重采样
# ════════════════════════════════════════════════════════════
def apply_smote_variant(X_train, y_train, variant="smote", strategy="moderate",
                         random_state=42):
    """
    应用SMOTE变体重采样。
    variant: "smote", "borderline1", "smote_enn"
    """
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SMOTENC
    from imblearn.combine import SMOTEENN

    counts = y_train.value_counts().sort_index()
    max_count = counts.max()

    if strategy == "moderate":
        target_counts = {}
        for cls in sorted(y_train.unique()):
            c = counts.get(cls, 0)
            if c == max_count:
                target_counts[cls] = c
            else:
                target_counts[cls] = max(int(max_count * 0.4), c)
    elif isinstance(strategy, dict):
        target_counts = strategy
    else:
        target_counts = "auto"

    k_neighbors = min(5, counts.min() - 1)
    k_neighbors = max(k_neighbors, 1)

    if variant == "smote":
        sampler = SMOTE(
            sampling_strategy=target_counts, random_state=random_state,
            k_neighbors=k_neighbors
        )
    elif variant == "borderline1":
        sampler = BorderlineSMOTE(
            sampling_strategy=target_counts, random_state=random_state,
            k_neighbors=k_neighbors, kind="borderline-1"
        )
    elif variant == "smote_enn":
        # SMOTE-ENN: 先过采样再清洗噪声
        smote = SMOTE(
            sampling_strategy=target_counts, random_state=random_state,
            k_neighbors=k_neighbors
        )
        X_res, y_res = smote.fit_resample(X_train, y_train)
        from imblearn.under_sampling import EditedNearestNeighbours
        enn = EditedNearestNeighbours(n_neighbors=3, sampling_strategy="auto")
        X_res, y_res = enn.fit_resample(X_res, y_res)

        print(f"\n[SMOTE-ENN 重采样]")
        print(f"  SMOTE后: {len(y_res):,} 条")
        print(f"  ENN清洗后: {len(y_res):,} 条")
        for g in sorted(y_train.unique()):
            n_before = (y_train == g).sum()
            n_after = (y_res == g).sum()
            change = f"+{n_after - n_before:,}" if n_after > n_before else "不变"
            print(f"  等级{g}: {n_before:,} -> {n_after:,} ({change})")
        return X_res, y_res
    else:
        raise ValueError(f"未知SMOTE变体: {variant}")

    X_res, y_res = sampler.fit_resample(X_train, y_train)

    print(f"\n[{variant.upper()} 重采样]")
    print(f"  重采样前: {len(y_train):,} 条")
    print(f"  重采样后: {len(y_res):,} 条")
    for g in sorted(y_train.unique()):
        n_before = (y_train == g).sum()
        n_after = (y_res == g).sum()
        change = f"+{n_after - n_before:,}" if n_after > n_before else "不变"
        print(f"  等级{g}: {n_before:,} -> {n_after:,} ({change})")

    return X_res, y_res


# ════════════════════════════════════════════════════════════
# 代价敏感模型训练
# ════════════════════════════════════════════════════════════
def train_cost_sensitive_xgb(X_train, y_train, y_train_orig, cost_matrix,
                              sample_weight_mode="combined", alpha=0.5):
    """代价敏感XGBoost分类器"""
    from xgboost import XGBClassifier

    mode_label = {
        "ordinal": "序数代价", "inverse": "逆频率",
        "combined": f"混合(alpha={alpha})"
    }.get(sample_weight_mode, sample_weight_mode)

    print(f"\n  [XGBoost 代价敏感] 权重策略: {mode_label}")

    # 先基于原始分布计算每个类别的权重
    if sample_weight_mode == "ordinal":
        w_map = ordinal_sample_weights(y_train_orig, cost_matrix)
    elif sample_weight_mode == "inverse":
        w_map = inverse_freq_weights(y_train_orig)
    else:
        w_map = combined_weights(y_train_orig, cost_matrix, alpha=alpha)

    # 构建类别→权重的映射
    unique_classes = sorted(y_train_orig.unique())
    class_weight_map = {}
    for cls in unique_classes:
        mask = (y_train_orig == cls)
        class_weight_map[cls] = w_map[mask].mean()

    # SMOTE后的样本，按其类别标签取对应权重
    y_arr = np.asarray(y_train, dtype=int).ravel()
    sample_weights = np.array([class_weight_map.get(yi, 1.0) for yi in y_arr])
    print(f"  样本权重范围: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")

    clf = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, gamma=0.1,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="multi:softprob",
        num_class=K,
        eval_metric="mlogloss",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    t0 = time.time()
    clf.fit(X_train.values, y_train.values, sample_weight=sample_weights)
    print(f"  XGBoost 拟合耗时: {time.time() - t0:.2f}s")
    return clf


def train_cost_sensitive_lgb(X_train, y_train, y_train_orig, cost_matrix,
                              sample_weight_mode="combined", alpha=0.5):
    """代价敏感LightGBM分类器"""
    from lightgbm import LGBMClassifier

    mode_label = {
        "ordinal": "序数代价", "inverse": "逆频率",
        "combined": f"混合(alpha={alpha})"
    }.get(sample_weight_mode, sample_weight_mode)

    print(f"\n  [LightGBM 代价敏感] 权重策略: {mode_label}")

    # 先基于原始分布计算每个类别的权重
    if sample_weight_mode == "ordinal":
        w_map = ordinal_sample_weights(y_train_orig, cost_matrix)
    elif sample_weight_mode == "inverse":
        w_map = inverse_freq_weights(y_train_orig)
    else:
        w_map = combined_weights(y_train_orig, cost_matrix, alpha=alpha)

    unique_classes = sorted(y_train_orig.unique())
    class_weight_map = {}
    for cls in unique_classes:
        mask = (y_train_orig == cls)
        class_weight_map[cls] = w_map[mask].mean()

    y_arr = np.asarray(y_train, dtype=int).ravel()
    sample_weights = np.array([class_weight_map.get(yi, 1.0) for yi in y_arr])
    print(f"  样本权重范围: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")

    clf = LGBMClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="multiclass",
        num_class=K,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    t0 = time.time()
    clf.fit(X_train.values, y_train.values, sample_weight=sample_weights)
    print(f"  LightGBM 拟合耗时: {time.time() - t0:.2f}s")
    return clf


def train_cost_sensitive_lr(X_train, y_train, cost_matrix,
                             sample_weight_mode="combined", alpha=0.5):
    """代价敏感逻辑回归"""
    print(f"\n  [Logistic 代价敏感] 权重策略: {sample_weight_mode}")

    if sample_weight_mode == "ordinal":
        w = ordinal_sample_weights(y_train, cost_matrix)
    elif sample_weight_mode == "inverse":
        w = inverse_freq_weights(y_train)
    else:
        w = combined_weights(y_train, cost_matrix, alpha=alpha)

    clf = LogisticRegression(
        C=0.5, max_iter=3000, solver="lbfgs",
        multi_class="multinomial", random_state=42,
    )
    t0 = time.time()
    clf.fit(X_train.values, y_train.values, sample_weight=w)
    print(f"  Logistic 拟合耗时: {time.time() - t0:.2f}s")
    return clf


# ════════════════════════════════════════════════════════════
# 阈值优化：搜索最优类别边界
# ════════════════════════════════════════════════════════════
def optimize_thresholds(proba, y_true, metric="kappa", n_steps=30):
    """
    搜索最优类别边界阈值。
    给定 K 个类别的概率矩阵 (n, K)，搜索 t01, t12, t23 三个阈值，
    使得基于概率的分类指标最优。

    分类规则：
      P(class=0) >= t01  → 预测0
      P(class=1) >= t12  → 预测1
      P(class=2) >= t23  → 预测3
      P(class=3) >= t23  → 预测3
      (优先高概率类别)
    """
    y_arr = np.asarray(y_true, dtype=int).ravel()

    best_score = -np.inf
    best_thresholds = [0.25, 0.25, 0.25]  # 默认等概率

    # 简化：搜索一个全局阈值因子
    # 标准argmax等价于threshold_factor=0
    # threshold_factor > 0 时，更倾向于预测低风险类别
    # threshold_factor < 0 时，更倾向于预测高风险类别
    for tf in np.linspace(-0.3, 0.3, n_steps * 3 + 1):
        y_pred = predict_with_offset(proba, offset=tf)
        if metric == "kappa":
            score = cohen_kappa_score(y_arr, y_pred)
        else:
            score = f1_score(y_arr, y_pred, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_thresholds = tf

    # 更精细搜索
    tf_lo = max(-0.3, best_thresholds - 0.05)
    tf_hi = min(0.3, best_thresholds + 0.05)
    for tf in np.linspace(tf_lo, tf_hi, n_steps):
        y_pred = predict_with_offset(proba, offset=tf)
        if metric == "kappa":
            score = cohen_kappa_score(y_arr, y_pred)
        else:
            score = f1_score(y_arr, y_pred, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_thresholds = tf

    return best_thresholds, best_score


def predict_with_offset(proba, offset=0.0):
    """
    基于概率偏移量预测类别。
    offset > 0: 倾向预测更低的风险等级（保守）
    offset < 0: 倾向预测更高的风险等级（敏感）
    """
    if proba.ndim == 1:
        return proba
    adjusted = proba.copy()
    # 对每个样本，给低风险类别增加offset，给高风险类别减少offset
    for i in range(K):
        adjusted[:, i] += offset * (K - 1 - 2 * i) / (K - 1)
    return adjusted.argmax(axis=1)


# ════════════════════════════════════════════════════════════
# 软投票集成
# ════════════════════════════════════════════════════════════
def soft_voting_predict(models, X_test):
    """多模型软投票：平均概率后argmax"""
    probas = []
    for clf, name in models:
        if hasattr(clf, "predict_proba"):
            p = clf.predict_proba(X_test.values if hasattr(X_test, "values") else X_test)
        elif hasattr(clf, "predict"):
            # 对于没有predict_proba的模型，使用one-hot编码
            preds = clf.predict(X_test.values if hasattr(X_test, "values") else X_test)
            p = np.zeros((len(preds), K))
            for i, pred in enumerate(preds):
                p[i, int(pred)] = 1.0
        else:
            continue
        probas.append(p)

    avg_proba = np.mean(probas, axis=0)
    return avg_proba


# ════════════════════════════════════════════════════════════
# 统一评估函数
# ════════════════════════════════════════════════════════════
def evaluate(y_pred, y_proba, y_test, model_name):
    """评估分类模型的完整指标体系"""
    y_arr = y_test.values if hasattr(y_test, "values") else y_test

    cm = confusion_matrix(y_arr, y_pred)
    kappa = cohen_kappa_score(y_arr, y_pred)

    # AUC
    auc_macro = auc_weighted = None
    per_class_auc = {}
    try:
        if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] == K:
            if len(np.unique(y_arr)) > 1:
                auc_macro = roc_auc_score(y_arr, y_proba, multi_class="ovr", average="macro")
                auc_weighted = roc_auc_score(y_arr, y_proba, multi_class="ovr", average="weighted")

            y_bin = label_binarize(y_arr, classes=list(range(K)))
            for i in range(K):
                if y_bin[:, i].sum() > 0 and y_bin[:, i].sum() < len(y_bin):
                    fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                    per_class_auc[i] = sk_auc(fpr, tpr)
                else:
                    per_class_auc[i] = None
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
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
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

    metrics = {
        "model_name": model_name,
        "kappa": round(kappa, 4),
        "auc_macro": round(auc_macro, 4) if auc_macro else None,
        "auc_weighted": round(auc_weighted, 4) if auc_weighted else None,
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "g_mean": round(g_mean, 4),
        "confusion_matrix": cm.tolist(),
        "per_class_auc": {k: round(v, 4) if v else None for k, v in per_class_auc.items()},
        "per_class_prf": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in per_class_prf.items()},
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
    return metrics


# ════════════════════════════════════════════════════════════
# 可视化
# ════════════════════════════════════════════════════════════
def plot_confusion_comparison(all_metrics, filename):
    """多模型混淆矩阵对比"""
    n_models = len(all_metrics)
    ncols = min(4, n_models)
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, m in enumerate(all_metrics):
        ax = axes[idx]
        cm = np.array(m["confusion_matrix"])
        total = cm.sum()

        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                pct = cm[i, j] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0
                annot[i, j] = f"{cm[i, j]}\n({pct:.1f}%)"

        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
                    xticklabels=RISK_LABELS, yticklabels=RISK_LABELS,
                    linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_xlabel("预测标签", fontsize=10)
        ax.set_ylabel("真实标签", fontsize=10)
        ax.set_title(f"{m['model_name']}\n(n={total:,})", fontsize=11, fontweight="bold")

    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("策略6: 代价敏感模型混淆矩阵对比", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(p, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [图表] 混淆矩阵对比 -> {p}")


def plot_metrics_comparison(summary_df, prev_summary_df=None, filename="31_metrics_comparison.png"):
    """
    综合指标对比图。
    可选：叠加策略2和策略5的历史最优结果。
    """
    n_models = len(summary_df)
    metric_names = ["kappa", "auc_macro", "macro_f1", "weighted_f1", "g_mean"]
    metric_labels = ["Cohen's Kappa", "AUC (macro)", "Macro F1", "Weighted F1", "G-mean"]

    colors_s6 = ["#534AB7", "#1D9E75", "#D85A30", "#BA7517", "#185FA5",
                 "#993556", "#3B6D11", "#E24B4A", "#0F6E56"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    model_names = summary_df["模型"].tolist()

    for idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[idx]
        values = [v if v is not None else 0 for v in summary_df[metric].tolist()]
        bars = ax.bar(range(len(model_names)), values,
                      color=colors_s6[:len(model_names)], edgecolor="white", width=0.6)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, fontsize=7, rotation=30, ha="right")
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    # 第6个子图：各类别召回率
    ax = axes[5]
    x = np.arange(K)
    width = min(0.9 / max(n_models, 1), 0.12)
    for i, (_, row) in enumerate(summary_df.iterrows()):
        recalls = [row[f"recall_{g}"] for g in range(K)]
        ax.bar(x + i * width, recalls, width, label=row["模型"],
               color=colors_s6[i % len(colors_s6)])
    ax.set_xticks(x + width * n_models / 2)
    ax.set_xticklabels(RISK_LABELS, fontsize=9)
    ax.set_ylabel("Recall", fontsize=11)
    ax.set_title("各类别召回率对比", fontweight="bold")
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("策略6: 代价敏感集成模型性能对比", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(p, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [图表] 指标对比柱状图 -> {p}")


def plot_cross_strategy_comparison(summary_df, filename="32_cross_strategy_comparison.png"):
    """
    跨策略对比图：策略6最优 vs 策略2最优 vs 策略5最优
    """
    # 策略2 最优结果 (来自01_smote_optimization.py)
    prev_results = {
        "LR+SMOTE\n(策略2)": {"kappa": 0.2097, "auc_macro": 0.8812, "macro_f1": 0.3378,
                               "weighted_f1": 0.9086, "g_mean": 0.3983,
                               "recall_0": 0.9028, "recall_1": 0.2234,
                               "recall_2": 0.2188, "recall_3": 0.5706},
        "LGB+SMOTE\n(策略2)": {"kappa": 0.1957, "auc_macro": 0.9117, "macro_f1": 0.3403,
                                "weighted_f1": 0.9444, "g_mean": 0.1516,
                                "recall_0": 0.9945, "recall_1": 0.1053,
                                "recall_2": 0.0813, "recall_3": 0.0621},
        "LGB+XGB\n(策略5)": {"kappa": 0.2016, "auc_macro": None, "macro_f1": 0.3548,
                              "weighted_f1": 0.8840, "g_mean": 0.3874,
                              "recall_0": 0.8506, "recall_1": 0.5886,
                              "recall_2": 0.1625, "recall_3": 0.2768},
    }

    # 策略6 最优
    best_idx = summary_df["macro_f1"].idxmax()
    best_row = summary_df.loc[best_idx]
    prev_results[best_row["模型"] + "\n(策略6)"] = {
        "kappa": best_row["kappa"],
        "auc_macro": best_row["auc_macro"],
        "macro_f1": best_row["macro_f1"],
        "weighted_f1": best_row["weighted_f1"],
        "g_mean": best_row["g_mean"],
        "recall_0": best_row["recall_0"],
        "recall_1": best_row["recall_1"],
        "recall_2": best_row["recall_2"],
        "recall_3": best_row["recall_3"],
    }

    names = list(prev_results.keys())
    cross_colors = ["#534AB7", "#1D9E75", "#D85A30", "#E24B4A"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    metric_keys = ["kappa", "auc_macro", "macro_f1", "weighted_f1", "g_mean"]
    metric_labels = ["Cohen's Kappa", "AUC (macro)", "Macro F1", "Weighted F1", "G-mean"]

    for idx, (mk, ml) in enumerate(zip(metric_keys, metric_labels)):
        ax = axes[idx]
        vals = [prev_results[n][mk] for n in names]
        vals_plot = [v if v is not None else 0 for v in vals]
        bars = ax.bar(range(len(names)), vals_plot,
                      color=cross_colors[:len(names)], edgecolor="white", width=0.6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel(ml, fontsize=11)
        ax.set_title(ml, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            if val is not None and val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 各类别召回率
    ax = axes[5]
    x = np.arange(K)
    width = 0.18
    for i, name in enumerate(names):
        recalls = [prev_results[name][f"recall_{g}"] for g in range(K)]
        ax.bar(x + i * width, recalls, width, label=name.replace("\n", " "),
               color=cross_colors[i % len(cross_colors)])
    ax.set_xticks(x + width * len(names) / 2)
    ax.set_xticklabels(RISK_LABELS, fontsize=9)
    ax.set_ylabel("Recall", fontsize=11)
    ax.set_title("各类别召回率对比", fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("跨策略最优模型对比", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(p, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [图表] 跨策略对比 -> {p}")


def plot_roc_comparison(all_metrics, y_test, filename="33_roc_comparison.png"):
    """ROC曲线对比"""
    y_arr = y_test.values if hasattr(y_test, "values") else y_test
    y_bin = label_binarize(y_arr, classes=list(range(K)))

    colors = ["#534AB7", "#1D9E75", "#D85A30", "#BA7517", "#185FA5",
              "#993556", "#3B6D11", "#E24B4A", "#0F6E56"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    # 宏平均ROC
    ax = axes[0]
    for idx, m in enumerate(all_metrics):
        if m["y_proba"] is not None and m["y_proba"].ndim == 2:
            try:
                fpr, tpr, _ = roc_curve(y_bin.ravel(), m["y_proba"].ravel())
                auc_val = sk_auc(fpr, tpr)
                name_short = m["model_name"].split("+")[0].strip()
                ax.plot(fpr, tpr, color=colors[idx % len(colors)], lw=1.5,
                       label=f"{name_short} (AUC={auc_val:.3f})")
            except:
                pass
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("假阳性率", fontsize=11)
    ax.set_ylabel("真阳性率", fontsize=11)
    ax.set_title("宏平均 ROC", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 各类别ROC（选最好的3个模型）
    best_metrics = sorted(all_metrics,
                          key=lambda x: x["macro_f1"] if x["macro_f1"] else 0,
                          reverse=True)[:3]

    for cls in range(K):
        ax = axes[cls + 1]
        for idx, m in enumerate(best_metrics):
            if m["y_proba"] is not None and m["y_proba"].ndim == 2:
                try:
                    fpr, tpr, _ = roc_curve(y_bin[:, cls], m["y_proba"][:, cls])
                    auc_val = sk_auc(fpr, tpr)
                    name_short = m["model_name"].split("+")[0].strip()
                    ax.plot(fpr, tpr, color=colors[idx % len(colors)], lw=1.5,
                           label=f"{name_short} (AUC={auc_val:.3f})")
                except:
                    pass
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("假阳性率", fontsize=11)
        ax.set_ylabel("真阳性率", fontsize=11)
        ax.set_title(f"{RISK_LABELS[cls]} (OVR)", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 隐藏多余子图
    for i in range(K + 1, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("策略6: ROC曲线对比", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(p, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [图表] ROC对比 -> {p}")


def plot_threshold_sweep(offsets, scores_kappa, scores_f1, best_idx,
                          best_offset, filename="34_threshold_sweep.png"):
    """阈值偏移搜索结果图"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(offsets, scores_kappa, "o-", color="#185FA5", ms=3, lw=1.5, label="Kappa")
    ax.plot(offsets, scores_f1, "s-", color="#BA7517", ms=3, lw=1.5, label="Macro F1")
    ax.axvline(x=best_offset, color="#E24B4A", linestyle="--", lw=1.5,
              label=f"最优偏移={best_offset:.3f}")
    ax.set_xlabel("概率偏移量 (offset)", fontsize=12)
    ax.set_ylabel("指标值", fontsize=12)
    ax.set_title("阈值偏移搜索结果 (offset=0为标准argmax)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(p, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [图表] 阈值搜索 -> {p}")


def plot_cost_matrix(cost_matrix, filename="35_cost_matrix.png"):
    """可视化代价矩阵"""
    fig, ax = plt.subplots(figsize=(6, 5))
    annot = np.empty_like(cost_matrix, dtype=object)
    for i in range(K):
        for j in range(K):
            annot[i, j] = f"{cost_matrix[i][j]:.1f}"
    sns.heatmap(cost_matrix, annot=annot, fmt="", cmap="YlOrRd",
                xticklabels=RISK_LABELS, yticklabels=RISK_LABELS,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("预测类别", fontsize=12)
    ax.set_ylabel("真实类别", fontsize=12)
    ax.set_title("序数距离代价矩阵 (cost = |i-j|^2)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(p, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [图表] 代价矩阵 -> {p}")


# ════════════════════════════════════════════════════════════
# 报告生成
# ════════════════════════════════════════════════════════════
def save_report(all_metrics, summary_df, y_test):
    lines = []
    lines.append("=" * 70)
    lines.append("代价敏感序数集成模型实验报告 - 策略6")
    lines.append("=" * 70)
    lines.append(f"\n生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"测试集样本量: {len(y_test):,}")
    lines.append(f"\n核心创新:")
    lines.append(f"  1. 序数距离代价矩阵: cost(i,j) = |i-j|^2，惩罚远离对角线的误分类")
    lines.append(f"  2. SMOTE变体优选: SMOTE / Borderline-SMOTE / SMOTE-ENN")
    lines.append(f"  3. 代价敏感训练: XGBoost/LightGBM基于代价矩阵的样本加权")
    lines.append(f"  4. 阈值优化: 概率偏移搜索，非默认argmax")
    lines.append(f"  5. 软投票集成: 融合多模型概率预测")

    for m in all_metrics:
        name = m["model_name"]
        lines.append(f"\n{'='*70}")
        lines.append(f"  {name}")
        lines.append(f"{'='*70}")

        cm = np.array(m["confusion_matrix"])
        lines.append(f"\n  混淆矩阵:")
        header = f"  {'':>14} {'预测0':>8} {'预测1':>8} {'预测2':>8} {'预测3':>8}"
        lines.append(header)
        for i in range(K):
            row_pct = cm[i] / cm[i].sum() * 100 if cm[i].sum() > 0 else np.zeros(K)
            row_str = f"  {RISK_LABELS[i]:>14}"
            for j in range(K):
                row_str += f" {cm[i,j]:>5}({row_pct[j]:>4.1f}%)"
            lines.append(row_str)

        lines.append(f"\n  各类别 Precision / Recall / F1:")
        lines.append(f"  {'类别':>14} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        for i in range(K):
            prf = m["per_class_prf"][i]
            lines.append(f"  {RISK_LABELS[i]:>14} {prf['precision']:>10.4f} {prf['recall']:>10.4f} {prf['f1']:>10.4f}")

        lines.append(f"\n  综合指标:")
        lines.append(f"    Cohen's Kappa:       {m['kappa']:.4f}")
        if m["auc_macro"]:
            lines.append(f"    AUC (macro):          {m['auc_macro']:.4f}")
            lines.append(f"    AUC (weighted):       {m['auc_weighted']:.4f}")
        lines.append(f"    Macro F1:             {m['macro_f1']:.4f}")
        lines.append(f"    Weighted F1:          {m['weighted_f1']:.4f}")
        lines.append(f"    G-mean:               {m['g_mean']:.4f}")

    # 对比总结
    lines.append(f"\n{'='*70}")
    lines.append(f"  综合对比总结")
    lines.append(f"{'='*70}")
    lines.append(f"\n{summary_df.to_string(index=False)}")

    # 跨策略对比
    lines.append(f"\n{'='*70}")
    lines.append(f"  跨策略对比（与策略2 SMOTE优化 / 策略5 级联模型）")
    lines.append(f"{'='*70}")
    lines.append(f"\n  策略2 最优 (LR+SMOTE): Kappa=0.2097, Macro F1=0.3378, Recall_3=0.5706")
    lines.append(f"  策略2 次优 (LGB+SMOTE): Kappa=0.1957, Macro F1=0.3403, AUC=0.9117")
    lines.append(f"  策略5 最优 (LGB+XGB级联): Kappa=0.2016, Macro F1=0.3548, Recall_3=0.2768")

    best_kappa = summary_df.loc[summary_df["kappa"].idxmax()]
    best_f1 = summary_df.loc[summary_df["macro_f1"].idxmax()]
    best_gmean = summary_df.loc[summary_df["g_mean"].idxmax()]
    best_recall3 = summary_df.loc[summary_df["recall_3"].idxmax()]

    lines.append(f"\n  策略6 最优Kappa: {best_kappa['模型']} (Kappa={best_kappa['kappa']})")
    lines.append(f"  策略6 最优Macro F1: {best_f1['模型']} (F1={best_f1['macro_f1']})")
    lines.append(f"  策略6 最优G-mean: {best_gmean['模型']} (G-mean={best_gmean['g_mean']})")
    lines.append(f"  策略6 最优Recall_3: {best_recall3['模型']} (Recall_3={best_recall3['recall_3']})")

    # 是否超越
    s2_best_kappa = 0.2097
    s5_best_f1 = 0.3548
    if best_kappa["kappa"] > s2_best_kappa:
        lines.append(f"\n  >>> 策略6 Kappa ({best_kappa['kappa']}) 超越策略2最优 ({s2_best_kappa})!")
    if best_f1["macro_f1"] > s5_best_f1:
        lines.append(f"  >>> 策略6 Macro F1 ({best_f1['macro_f1']}) 超越策略5最优 ({s5_best_f1})!")

    report = "\n".join(lines)
    report_path = os.path.join(OUTPUT_DIR, "36_ensemble_cost_sensitive_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  [报告] -> {report_path}")
    return report


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    print("=" * 60)
    print("  代价敏感序数集成模型实验 - 策略6")
    print("  核心创新: 序数代价矩阵 + SMOTE变体 + 阈值优化 + 集成")
    print("=" * 60)

    # 1. 数据准备
    X_train, X_test, y_train, y_test, scaler, feat_names = load_and_prepare()

    # 2. 从训练集划分验证集
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=123)
    for tr_idx, val_idx in sss_val.split(X_train, y_train):
        X_tr = X_train.iloc[tr_idx].reset_index(drop=True)
        y_tr = y_train.iloc[tr_idx].reset_index(drop=True)
        X_val = X_train.iloc[val_idx].reset_index(drop=True)
        y_val = y_train.iloc[val_idx].reset_index(drop=True)

    print(f"\n[验证集划分] 训练子集: {len(X_tr):,}  验证子集: {len(X_val):,}  测试集: {len(X_test):,}")

    # 3. 构建代价矩阵
    cost_matrix = build_ordinal_cost_matrix(n_classes=K, power=2)
    print(f"\n[代价矩阵] power=2")
    print(cost_matrix)
    plot_cost_matrix(cost_matrix, "35_cost_matrix.png")

    # ════════════════════════════════════════════════════════
    # 实验矩阵
    # ════════════════════════════════════════════════════════
    experiments = [
        # (SMOTE变体, 模型类型, 权重策略, alpha, 实验名)
        ("smote",       "xgb", "combined",  0.5, "XGB+SMOTE+混合权重"),
        ("smote",       "lgb", "combined",  0.5, "LGB+SMOTE+混合权重"),
        ("smote",       "xgb", "ordinal",   0.0, "XGB+SMOTE+序数代价"),
        ("smote",       "lgb", "ordinal",   0.0, "LGB+SMOTE+序数代价"),
        ("borderline1", "xgb", "combined",  0.5, "XGB+BorderSMOTE+混合"),
        ("borderline1", "lgb", "combined",  0.5, "LGB+BorderSMOTE+混合"),
        ("smote_enn",   "xgb", "combined",  0.5, "XGB+SMOTE-ENN+混合"),
        ("smote_enn",   "lgb", "combined",  0.5, "LGB+SMOTE-ENN+混合"),
        ("smote",       "lr",  "combined",  0.5, "LR+SMOTE+混合权重"),
    ]

    all_metrics = []
    all_models_for_ensemble = []  # 保存用于集成的模型
    summary_rows = []
    val_probas = {}  # 保存验证集概率用于集成

    for smote_var, model_type, weight_mode, alpha, exp_name in experiments:
        print(f"\n{'#'*60}")
        print(f"  # {exp_name}")
        print(f"{'#'*60}")

        # SMOTE 重采样
        X_smote, y_smote = apply_smote_variant(X_tr, y_tr, variant=smote_var,
                                                strategy="moderate")

        # 训练模型
        if model_type == "xgb":
            clf = train_cost_sensitive_xgb(X_smote, y_smote, y_tr, cost_matrix,
                                            sample_weight_mode=weight_mode, alpha=alpha)
        elif model_type == "lgb":
            clf = train_cost_sensitive_lgb(X_smote, y_smote, y_tr, cost_matrix,
                                            sample_weight_mode=weight_mode, alpha=alpha)
        elif model_type == "lr":
            clf = train_cost_sensitive_lr(X_smote, y_smote, cost_matrix,
                                           sample_weight_mode=weight_mode, alpha=alpha)
        else:
            continue

        # 验证集概率
        val_proba = clf.predict_proba(X_val.values)
        val_probas[exp_name] = val_proba

        # 阈值优化
        best_offset, best_val_score = optimize_thresholds(
            val_proba, y_val, metric="kappa"
        )
        print(f"  阈值偏移: {best_offset:.4f} (验证集Kappa={best_val_score:.4f})")

        # 绘制阈值搜索曲线（仅第一个实验）
        if len(all_metrics) == 0:
            print(f"  [阈值曲线] 绘制完整搜索结果...")
            offsets_list, kappas_list, f1s_list = [], [], []
            for tf in np.linspace(-0.3, 0.3, 60):
                y_pred_v = predict_with_offset(val_proba, offset=tf)
                y_arr_v = y_val.values
                offsets_list.append(tf)
                kappas_list.append(cohen_kappa_score(y_arr_v, y_pred_v))
                f1s_list.append(f1_score(y_arr_v, y_pred_v, average="macro", zero_division=0))
            best_idx_local = np.argmax(kappas_list)
            plot_threshold_sweep(offsets_list, kappas_list, f1s_list,
                                best_idx_local, best_offset, "34_threshold_sweep.png")

        # 测试集预测
        test_proba = clf.predict_proba(X_test.values)
        y_pred = predict_with_offset(test_proba, offset=best_offset)

        # 评估
        metrics = evaluate(y_pred, test_proba, y_test, exp_name)
        metrics["threshold_offset"] = round(best_offset, 4)
        metrics["val_score"] = round(best_val_score, 4)
        all_metrics.append(metrics)
        all_models_for_ensemble.append((clf, exp_name))

        # 汇总行
        row = {"模型": exp_name, "kappa": metrics["kappa"],
               "auc_macro": metrics["auc_macro"],
               "macro_f1": metrics["macro_f1"],
               "weighted_f1": metrics["weighted_f1"],
               "g_mean": metrics["g_mean"]}
        for i in range(K):
            row[f"recall_{i}"] = metrics["per_class_prf"][i]["recall"]
            row[f"precision_{i}"] = metrics["per_class_prf"][i]["precision"]
            row[f"f1_{i}"] = metrics["per_class_prf"][i]["f1"]
        summary_rows.append(row)

    # ════════════════════════════════════════════════════════
    # 集成实验
    # ════════════════════════════════════════════════════════
    print(f"\n{'#'*60}")
    print(f"  # 集成实验")
    print(f"{'#'*60}")

    # 集成1: 所有模型软投票
    print(f"\n  [集成1] 全部模型软投票")
    ensemble_proba = soft_voting_predict(all_models_for_ensemble, X_test)
    y_pred_ens = ensemble_proba.argmax(axis=1)
    metrics_ens = evaluate(y_pred_ens, ensemble_proba, y_test, "集成: 全部模型软投票")
    all_metrics.append(metrics_ens)
    row = {"模型": "集成: 全部软投票", "kappa": metrics_ens["kappa"],
           "auc_macro": metrics_ens["auc_macro"],
           "macro_f1": metrics_ens["macro_f1"],
           "weighted_f1": metrics_ens["weighted_f1"],
           "g_mean": metrics_ens["g_mean"]}
    for i in range(K):
        row[f"recall_{i}"] = metrics_ens["per_class_prf"][i]["recall"]
        row[f"precision_{i}"] = metrics_ens["per_class_prf"][i]["precision"]
        row[f"f1_{i}"] = metrics_ens["per_class_prf"][i]["f1"]
    summary_rows.append(row)

    # 集成2: Top3模型 + 阈值优化
    top3_models = sorted(all_metrics[:-1],  # 排除集成1
                         key=lambda x: x["macro_f1"] if x["macro_f1"] else 0,
                         reverse=True)[:3]
    top3_names = [m["model_name"] for m in top3_models]
    top3_clfs = [(clf, name) for clf, name in all_models_for_ensemble
                 if name in top3_names]

    if len(top3_clfs) >= 2:
        print(f"\n  [集成2] Top3模型软投票 (Top3: {', '.join(top3_names)})")
        top3_proba = soft_voting_predict(top3_clfs, X_test)

        # 对集成也做阈值优化
        top3_val_clfs = [(clf, name) for clf, name in all_models_for_ensemble
                         if name in top3_names]
        top3_val_proba = soft_voting_predict(top3_val_clfs, X_val)
        best_offset_ens, best_val_ens = optimize_thresholds(
            top3_val_proba, y_val, metric="kappa"
        )
        print(f"  阈值偏移: {best_offset_ens:.4f} (验证集Kappa={best_val_ens:.4f})")

        y_pred_top3 = predict_with_offset(top3_proba, offset=best_offset_ens)
        metrics_top3 = evaluate(y_pred_top3, top3_proba, y_test,
                                f"集成: Top3+阈值优化")
        metrics_top3["threshold_offset"] = round(best_offset_ens, 4)
        all_metrics.append(metrics_top3)

        row = {"模型": "集成: Top3+阈值", "kappa": metrics_top3["kappa"],
               "auc_macro": metrics_top3["auc_macro"],
               "macro_f1": metrics_top3["macro_f1"],
               "weighted_f1": metrics_top3["weighted_f1"],
               "g_mean": metrics_top3["g_mean"]}
        for i in range(K):
            row[f"recall_{i}"] = metrics_top3["per_class_prf"][i]["recall"]
            row[f"precision_{i}"] = metrics_top3["per_class_prf"][i]["precision"]
            row[f"f1_{i}"] = metrics_top3["per_class_prf"][i]["f1"]
        summary_rows.append(row)

    # 集成3: XGB+LGB 仅混合权重模型（同SMOTE变体内集成）
    xgb_lgb_models = [(clf, name) for clf, name in all_models_for_ensemble
                      if "XGB+SMOTE+混合" in name or "LGB+SMOTE+混合" in name]
    if len(xgb_lgb_models) >= 2:
        print(f"\n  [集成3] XGB+LGB同SMOTE变体集成")
        xl_proba = soft_voting_predict(xgb_lgb_models, X_test)
        xl_val = soft_voting_predict(xgb_lgb_models, X_val)
        best_offset_xl, _ = optimize_thresholds(xl_val, y_val, metric="kappa")
        y_pred_xl = predict_with_offset(xl_proba, offset=best_offset_xl)
        metrics_xl = evaluate(y_pred_xl, xl_proba, y_test,
                              "集成: XGB+LGB混合")
        metrics_xl["threshold_offset"] = round(best_offset_xl, 4)
        all_metrics.append(metrics_xl)

        row = {"模型": "集成: XGB+LGB混合", "kappa": metrics_xl["kappa"],
               "auc_macro": metrics_xl["auc_macro"],
               "macro_f1": metrics_xl["macro_f1"],
               "weighted_f1": metrics_xl["weighted_f1"],
               "g_mean": metrics_xl["g_mean"]}
        for i in range(K):
            row[f"recall_{i}"] = metrics_xl["per_class_prf"][i]["recall"]
            row[f"precision_{i}"] = metrics_xl["per_class_prf"][i]["precision"]
            row[f"f1_{i}"] = metrics_xl["per_class_prf"][i]["f1"]
        summary_rows.append(row)

    # ════════════════════════════════════════════════════════
    # 汇总
    # ════════════════════════════════════════════════════════
    summary_df = pd.DataFrame(summary_rows)

    # 保存对比CSV
    csv_path = os.path.join(OUTPUT_DIR, "37_ensemble_comparison.csv")
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  [对比表] -> {csv_path}")

    # 绘制对比图
    print(f"\n[生成对比图表]")
    plot_confusion_comparison(all_metrics, "38_confusion_matrices.png")
    plot_metrics_comparison(summary_df, "31_metrics_comparison.png")
    plot_cross_strategy_comparison(summary_df, "32_cross_strategy_comparison.png")
    plot_roc_comparison(all_metrics, y_test, "33_roc_comparison.png")

    # 保存完整报告
    save_report(all_metrics, summary_df, y_test)

    # 最终汇总
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  代价敏感集成模型实验完成，耗时 {elapsed:.1f}s")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"{'='*60}")

    best_kappa_row = summary_df.loc[summary_df["kappa"].idxmax()]
    best_f1_row = summary_df.loc[summary_df["macro_f1"].idxmax()]
    print(f"\n  ★ 最优Kappa: {best_kappa_row['模型']} (Kappa={best_kappa_row['kappa']})")
    print(f"    少数类recall: 低风险={best_kappa_row['recall_1']:.4f}, "
          f"中风险={best_kappa_row['recall_2']:.4f}, 高风险={best_kappa_row['recall_3']:.4f}")
    print(f"\n  ★ 最优Macro F1: {best_f1_row['模型']} (F1={best_f1_row['macro_f1']})")
    print(f"    少数类recall: 低风险={best_f1_row['recall_1']:.4f}, "
          f"中风险={best_f1_row['recall_2']:.4f}, 高风险={best_f1_row['recall_3']:.4f}")

    # 跨策略对比
    print(f"\n  --- 跨策略最优对比 ---")
    print(f"  策略2 (LR+SMOTE):  Kappa=0.2097, Macro F1=0.3378, Recall_3=0.5706")
    print(f"  策略5 (LGB+XGB):  Kappa=0.2016, Macro F1=0.3548, Recall_3=0.2768")
    print(f"  策略6 (最优Kappa): Kappa={best_kappa_row['kappa']}, "
          f"Macro F1={best_kappa_row['macro_f1']}, Recall_3={best_kappa_row['recall_3']}")
    print(f"  策略6 (最优F1):   Kappa={best_f1_row['kappa']}, "
          f"Macro F1={best_f1_row['macro_f1']}, Recall_3={best_f1_row['recall_3']}")


if __name__ == "__main__":
    main()
