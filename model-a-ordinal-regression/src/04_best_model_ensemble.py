"""
肿瘤患者压力性损伤风险预测 —— 最优模型（综合版）
====================================================
方案: Top3软投票集成 + 阈值优化
基模型: LGB+BorderSMOTE+混合权重 / LR+SMOTE+混合权重 / LGB+SMOTE+混合权重
验证性能: Kappa=0.2788, Macro F1=0.3859, G-mean=0.3329

特点:
  - 序数距离代价矩阵 (cost=|i-j|^2) 驱动的混合样本权重
  - Borderline-SMOTE / SMOTE 过采样
  - 验证集概率偏移阈值优化
  - 三模型软投票集成

环境要求: Python 3.8+, xgboost, lightgbm, imbalanced-learn, scikit-learn
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
from sklearn.linear_model import LogisticRegression
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

# 最优阈值偏移（验证集搜索得到，此处固定）
# 如果需要重新搜索，设为 None
FIXED_THRESHOLD_OFFSET = None


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
    """基于代价矩阵为每个样本计算权重"""
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


def inverse_freq_weights(y):
    """逆频率权重"""
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
    """混合权重 = alpha * 逆频率 + (1-alpha) * 序数代价"""
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
# SMOTE 重采样
# ════════════════════════════════════════════════════════════
def apply_smote_variant(X_train, y_train, variant="smote", random_state=42):
    """
    variant: "smote" 或 "borderline1"
    使用 moderate 策略: 少数类上采样到多数类的40%
    """
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE

    counts = y_train.value_counts().sort_index()
    max_count = counts.max()
    target_counts = {}
    for cls in sorted(y_train.unique()):
        c = counts.get(cls, 0)
        target_counts[cls] = max(int(max_count * 0.4), c) if c < max_count else c

    k_neighbors = max(min(5, counts.min() - 1), 1)

    if variant == "borderline1":
        sampler = BorderlineSMOTE(
            sampling_strategy=target_counts, random_state=random_state,
            k_neighbors=k_neighbors, kind="borderline-1"
        )
    else:
        sampler = SMOTE(
            sampling_strategy=target_counts, random_state=random_state,
            k_neighbors=k_neighbors
        )

    X_res, y_res = sampler.fit_resample(X_train, y_train)
    print(f"  [{variant.upper()}] {len(y_train):,} -> {len(y_res):,} 条")
    return X_res, y_res


# ════════════════════════════════════════════════════════════
# 模型训练
# ════════════════════════════════════════════════════════════
def get_class_weights(y_train_orig, cost_matrix, alpha=0.5):
    """基于原始训练集分布计算类别权重映射"""
    w_map = combined_weights(y_train_orig, cost_matrix, alpha=alpha)
    unique_classes = sorted(y_train_orig.unique())
    class_weight_map = {}
    for cls in unique_classes:
        mask = (y_train_orig == cls)
        class_weight_map[cls] = w_map[mask].mean()
    return class_weight_map


def train_lgb_borderline(X_tr, y_tr, cost_matrix):
    """基模型1: LightGBM + Borderline-SMOTE + 混合权重"""
    from lightgbm import LGBMClassifier

    print("  [基模型1] LightGBM + Borderline-SMOTE + 混合权重")
    X_smote, y_smote = apply_smote_variant(X_tr, y_tr, variant="borderline1")
    cw = get_class_weights(y_tr, cost_matrix, alpha=0.5)

    y_arr = np.asarray(y_smote, dtype=int).ravel()
    sample_weights = np.array([cw.get(yi, 1.0) for yi in y_arr])

    clf = LGBMClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="multiclass", num_class=K,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    clf.fit(X_smote.values, y_smote.values, sample_weight=sample_weights)
    print(f"    拟合完成")
    return clf


def train_lr_smote(X_tr, y_tr, cost_matrix):
    """基模型2: Logistic Regression + SMOTE + 混合权重"""
    print("  [基模型2] Logistic Regression + SMOTE + 混合权重")
    X_smote, y_smote = apply_smote_variant(X_tr, y_tr, variant="smote")
    w = combined_weights(y_smote, cost_matrix, alpha=0.5)

    clf = LogisticRegression(
        C=0.5, max_iter=3000, solver="lbfgs",
        multi_class="multinomial", random_state=42,
    )
    clf.fit(X_smote.values, y_smote.values, sample_weight=w)
    print(f"    拟合完成")
    return clf


def train_lgb_smote(X_tr, y_tr, cost_matrix):
    """基模型3: LightGBM + SMOTE + 混合权重"""
    from lightgbm import LGBMClassifier

    print("  [基模型3] LightGBM + SMOTE + 混合权重")
    X_smote, y_smote = apply_smote_variant(X_tr, y_tr, variant="smote")
    cw = get_class_weights(y_tr, cost_matrix, alpha=0.5)

    y_arr = np.asarray(y_smote, dtype=int).ravel()
    sample_weights = np.array([cw.get(yi, 1.0) for yi in y_arr])

    clf = LGBMClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="multiclass", num_class=K,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    clf.fit(X_smote.values, y_smote.values, sample_weight=sample_weights)
    print(f"    拟合完成")
    return clf


# ════════════════════════════════════════════════════════════
# 阈值优化
# ════════════════════════════════════════════════════════════
def predict_with_offset(proba, offset=0.0):
    """基于概率偏移量预测类别"""
    adjusted = proba.copy()
    for i in range(K):
        adjusted[:, i] += offset * (K - 1 - 2 * i) / (K - 1)
    return adjusted.argmax(axis=1)


def optimize_threshold(proba, y_true, metric="kappa", n_steps=60):
    """搜索最优概率偏移"""
    y_arr = np.asarray(y_true, dtype=int).ravel()
    best_score = -np.inf
    best_offset = 0.0

    # 粗搜索
    for tf in np.linspace(-0.3, 0.3, n_steps * 2 + 1):
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
# 软投票集成
# ════════════════════════════════════════════════════════════
def ensemble_predict_proba(models, X):
    """多模型软投票：平均概率"""
    probas = []
    for clf in models:
        p = clf.predict_proba(X.values if hasattr(X, "values") else X)
        probas.append(p)
    return np.mean(probas, axis=0)


# ════════════════════════════════════════════════════════════
# 评估
# ════════════════════════════════════════════════════════════
def evaluate_model(y_pred, y_proba, y_test, model_name="最优集成模型"):
    """完整评估并打印结果"""
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

    print(f"\n  sklearn classification_report:")
    print(classification_report(y_arr, y_pred, target_names=RISK_LABELS, zero_division=0))

    return {
        "model_name": model_name,
        "kappa": kappa, "auc_macro": auc_macro, "auc_weighted": auc_weighted,
        "macro_f1": macro_f1, "weighted_f1": weighted_f1, "g_mean": g_mean,
        "confusion_matrix": cm, "per_class_prf": per_class_prf,
        "per_class_auc": per_class_auc,
    }


def plot_results(metrics, y_test, y_proba, filename_prefix="best_model"):
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
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
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
            ax.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f"{RISK_LABELS[i]} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("假阳性率", fontsize=11)
    ax.set_ylabel("真阳性率", fontsize=11)
    ax.set_title("ROC曲线 (OVR)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"最优集成模型评估 (Kappa={metrics['kappa']:.4f}, Macro F1={metrics['macro_f1']:.4f})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, f"{filename_prefix}_evaluation.png")
    plt.savefig(p, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [图表] 评估图 -> {p}")


# ════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════
def train_and_evaluate():
    """完整的训练+评估流程"""
    t_start = time.time()
    print("=" * 65)
    print("  肿瘤患者压力性损伤风险预测 - 最优集成模型")
    print("  方案: Top3软投票 + 阈值优化")
    print("=" * 65)

    # 1. 数据准备
    print("\n[步骤1] 数据准备")
    X_train, X_test, y_train, y_test, scaler, feat_names = load_and_prepare()

    # 2. 划分训练/验证集
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

    # 4. 训练三个基模型
    print("\n[步骤4] 训练基模型")
    model1 = train_lgb_borderline(X_tr, y_tr, cost_matrix)
    model2 = train_lr_smote(X_tr, y_tr, cost_matrix)
    model3 = train_lgb_smote(X_tr, y_tr, cost_matrix)
    models = [model1, model2, model3]

    # 5. 集成 + 阈值优化
    print("\n[步骤5] 集成预测 + 阈值优化")
    val_proba = ensemble_predict_proba(models, X_val)

    if FIXED_THRESHOLD_OFFSET is not None:
        best_offset = FIXED_THRESHOLD_OFFSET
        y_val_pred = predict_with_offset(val_proba, offset=best_offset)
        best_val_kappa = cohen_kappa_score(y_val.values, y_val_pred)
        print(f"  使用固定阈值偏移: {best_offset:.4f}")
        print(f"  验证集Kappa: {best_val_kappa:.4f}")
    else:
        best_offset, best_val_kappa = optimize_threshold(val_proba, y_val, metric="kappa")
        print(f"  最优阈值偏移: {best_offset:.4f}")
        print(f"  验证集Kappa: {best_val_kappa:.4f}")

    # 6. 测试集评估
    print("\n[步骤6] 测试集评估")
    test_proba = ensemble_predict_proba(models, X_test)
    y_pred = predict_with_offset(test_proba, offset=best_offset)
    metrics = evaluate_model(y_pred, test_proba, y_test)

    # 7. 保存图表
    print("\n[步骤7] 生成评估图表")
    plot_results(metrics, y_test, test_proba, "best_ensemble_model")

    # 8. 保存模型
    import joblib
    model_pkg = {
        "models": models,
        "scaler": scaler,
        "feat_names": feat_names,
        "threshold_offset": best_offset,
        "cost_matrix": cost_matrix,
        "risk_labels": RISK_LABELS,
    }
    model_path = os.path.join(MODEL_DIR, "best_ensemble_model.pkl")
    joblib.dump(model_pkg, model_path)
    print(f"  [模型] 已保存 -> {model_path}")

    # 汇总
    elapsed = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"  完成! 总耗时 {elapsed:.1f}s")
    print(f"  Kappa={metrics['kappa']:.4f}, Macro F1={metrics['macro_f1']:.4f}, G-mean={metrics['g_mean']:.4f}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  模型文件: {model_path}")
    print(f"{'='*65}")


if __name__ == "__main__":
    train_and_evaluate()
