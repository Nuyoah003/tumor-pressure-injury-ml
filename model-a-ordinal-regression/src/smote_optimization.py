"""
肿瘤患者压力性损伤风险预测模型 —— SMOTE优化实验（策略2）
=================================================================
核心策略：保留4类分类 + SMOTE重采样 + 类别权重 + 树模型

对比模型：
  1. 有序逻辑回归（原始基线，无SMOTE）
  2. 有序逻辑回归 + SMOTE
  3. XGBoost + SMOTE + 类别权重
  4. LightGBM + SMOTE + 类别权重

评估指标：
  - 混淆矩阵（含百分比）
  - 多分类 AUC (macro / weighted / per-class)
  - Cohen's Kappa
  - ROC 曲线
  - Macro F1-score
  - G-mean（每类召回率的几何平均）
  - 各类别 Precision / Recall / F1
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats as sp_stats
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

# ── 路径配置 ──
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "清洗后数据.xlsx")
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
# 数据准备（复用主流程逻辑）
# ════════════════════════════════════════════════════════════
def load_and_prepare():
    """加载数据、编码、划分、标准化，返回原始数据集（不SMOTE）"""
    df = pd.read_excel(DATA_PATH)
    print(f"[数据加载] {df.shape[0]:,} 行 x {df.shape[1]} 列")

    # 编码
    y = df[TARGET].astype(int)
    X = df.drop(columns=["ID", TARGET], errors="ignore")

    for col in NOMINAL_VARS:
        if col not in X.columns:
            continue
        dummies = pd.get_dummies(X[col], prefix=col, drop_first=False, dtype=int)
        dummies = dummies.iloc[:, :-1]  # 去掉最后一类（参照组）
        X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
    X = X.astype(float)
    feat_names = list(X.columns)

    # 分层划分
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for tr, te in sss.split(X, y):
        X_train = X.iloc[tr].reset_index(drop=True)
        X_test = X.iloc[te].reset_index(drop=True)
        y_train = y.iloc[tr].reset_index(drop=True)
        y_test = y.iloc[te].reset_index(drop=True)

    # 标准化
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
def apply_smote(X_train, y_train, strategy="auto", random_state=42):
    """
    应用SMOTE重采样训练集。
    strategy: 可选 "auto"（各类到多数类数量）、"moderate"（中间策略）、
              或 dict {0: n0, 1: n1, ...} 自定义目标数量
    """
    from imblearn.over_sampling import SMOTE

    if strategy == "moderate":
        # 中等策略：少数类增采样到多数类的30%~50%，避免过度合成
        counts = y_train.value_counts().sort_index()
        max_count = counts.max()
        target_counts = {}
        for cls in range(K):
            c = counts.get(cls, 0)
            if c == max_count:
                target_counts[cls] = c
            else:
                # 增采样到 max_count * 0.4（可调节）
                target = max(int(max_count * 0.4), c)
                target_counts[cls] = target
        smote = SMOTE(sampling_strategy=target_counts, random_state=random_state, k_neighbors=5)
    elif isinstance(strategy, dict):
        smote = SMOTE(sampling_strategy=strategy, random_state=random_state, k_neighbors=5)
    else:
        # auto: 全部增采样到多数类数量
        smote = SMOTE(sampling_strategy="auto", random_state=random_state, k_neighbors=5)

    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"\n[SMOTE 重采样]")
    print(f"  重采样前: {len(y_train):,} 条")
    print(f"  重采样后: {len(y_res):,} 条")
    for g in range(K):
        n_before = (y_train == g).sum()
        n_after = (y_res == g).sum()
        change = f"+{n_after - n_before:,}" if n_after > n_before else "不变"
        print(f"  {RISK_LABELS[g]}: {n_before:,} -> {n_after:,} ({change})")

    return X_res, y_res


# ════════════════════════════════════════════════════════════
# 模型1: 有序逻辑回归（原始基线）
# ════════════════════════════════════════════════════════════
def train_ordered_logit(X_train, y_train):
    """statsmodels OrderedModel（无SMOTE基线）"""
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    print("\n[模型1] 有序逻辑回归（原始基线，无SMOTE）")
    sm = OrderedModel(y_train.values, X_train.values, distr="logit")
    t0 = time.time()
    result = sm.fit(method="bfgs", disp=False, maxiter=5000)
    elapsed = time.time() - t0
    print(f"  拟合耗时: {elapsed:.2f}s")
    return result, "ordered_logit"


# ════════════════════════════════════════════════════════════
# 模型2: 有序逻辑回归 + SMOTE
# ════════════════════════════════════════════════════════════
def train_ordered_logit_smote(X_train_smote, y_train_smote):
    """statsmodels OrderedModel（SMOTE训练集）"""
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    print("\n[模型2] 有序逻辑回归 + SMOTE")
    sm = OrderedModel(y_train_smote.values, X_train_smote.values, distr="logit")
    t0 = time.time()
    result = sm.fit(method="bfgs", disp=False, maxiter=5000)
    elapsed = time.time() - t0
    print(f"  拟合耗时: {elapsed:.2f}s")
    return result, "ordered_logit_smote"


# ════════════════════════════════════════════════════════════
# 模型3: XGBoost + SMOTE + 类别权重
# ════════════════════════════════════════════════════════════
def train_xgboost(X_train_smote, y_train_smote, y_train_orig):
    """
    XGBoost 多分类模型。
    使用 SMOTE 训练集 + 样本权重（基于原始类别分布的逆频率）。
    """
    from xgboost import XGBClassifier

    print("\n[模型3] XGBoost + SMOTE + 类别权重")
    # 计算原始训练集的类别权重
    counts = y_train_orig.value_counts().sort_index()
    n_total = counts.sum()
    class_weights = {cls: n_total / (K * cnt) for cls, cnt in counts.items()}
    print(f"  类别权重: { {RISK_LABELS[k]: f'{v:.2f}' for k, v in class_weights.items()} }")

    # 构建样本权重（SMOTE后的每条样本取其类别的权重）
    sample_weights = np.array([class_weights[yi] for yi in y_train_smote.values])

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=K,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    t0 = time.time()
    model.fit(
        X_train_smote.values, y_train_smote.values,
        sample_weight=sample_weights,
    )
    elapsed = time.time() - t0
    print(f"  拟合耗时: {elapsed:.2f}s")
    return model, "xgboost"


# ════════════════════════════════════════════════════════════
# 模型4: LightGBM + SMOTE + 类别权重
# ════════════════════════════════════════════════════════════
def train_lightgbm(X_train_smote, y_train_smote, y_train_orig):
    """
    LightGBM 多分类模型。
    使用 SMOTE 训练集 + is_unbalance 处理不平衡。
    """
    from lightgbm import LGBMClassifier

    print("\n[模型4] LightGBM + SMOTE + 类别权重")

    model = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multiclass",
        num_class=K,
        is_unbalance=True,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    t0 = time.time()
    model.fit(X_train_smote.values, y_train_smote.values)
    elapsed = time.time() - t0
    print(f"  拟合耗时: {elapsed:.2f}s")
    return model, "lightgbm"


# ════════════════════════════════════════════════════════════
# 统一评估函数
# ════════════════════════════════════════════════════════════
def evaluate(model, model_type, X_test, y_test, model_name):
    """
    统一评估接口。
    返回: dict 包含所有指标
    """
    X_arr = X_test.values if hasattr(X_test, "values") else X_test
    y_arr = y_test.values if hasattr(y_test, "values") else y_test

    # 预测概率和类别
    if model_type in ("ordered_logit", "ordered_logit_smote"):
        proba = model.predict(X_arr)
        y_pred = proba.argmax(axis=1)
    else:
        proba = model.predict_proba(X_arr)
        y_pred = model.predict(X_arr)

    # 基础指标
    cm = confusion_matrix(y_arr, y_pred)
    kappa = cohen_kappa_score(y_arr, y_pred)

    # AUC
    try:
        auc_macro = roc_auc_score(y_arr, proba, multi_class="ovr", average="macro")
        auc_weighted = roc_auc_score(y_arr, proba, multi_class="ovr", average="weighted")
        # 各类别AUC
        y_bin = label_binarize(y_arr, classes=list(range(K)))
        per_class_auc = {}
        for i in range(K):
            if y_bin[:, i].sum() > 0:
                fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
                per_class_auc[i] = sk_auc(fpr, tpr)
            else:
                per_class_auc[i] = None
    except Exception as e:
        print(f"  AUC 计算跳过: {e}")
        auc_macro = auc_weighted = None
        per_class_auc = {}

    # Macro F1
    macro_f1 = f1_score(y_arr, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_arr, y_pred, average="weighted", zero_division=0)

    # 各类别 Precision / Recall / F1
    per_class_prf = {}
    for i in range(K):
        per_class_prf[i] = {
            "precision": precision_score(y_arr == i, y_pred == i, zero_division=0),
            "recall": recall_score(y_arr == i, y_pred == i, zero_division=0),
            "f1": f1_score(y_arr == i, y_pred == i, zero_division=0),
        }

    # G-mean: 各类别召回率的几何平均
    recalls = [per_class_prf[i]["recall"] for i in range(K)]
    # 过滤掉召回率为0的类（避免log(0)）
    recalls_pos = [r for r in recalls if r > 0]
    g_mean = np.exp(np.mean(np.log(recalls_pos))) if recalls_pos else 0.0

    # 汇总
    metrics = {
        "model_name": model_name,
        "model_type": model_type,
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
        "y_proba": proba,
    }
    return metrics


def print_evaluation(metrics, y_test):
    """打印单个模型的评估结果"""
    name = metrics["model_name"]
    y_arr = y_test.values if hasattr(y_test, "values") else y_test

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    cm = np.array(metrics["confusion_matrix"])
    n = cm.sum()

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
        prf = metrics["per_class_prf"][i]
        print(f"  {RISK_LABELS[i]:>12} {prf['precision']:>10.4f} {prf['recall']:>10.4f} {prf['f1']:>10.4f}")

    print(f"\n  综合指标:")
    print(f"  Cohen's Kappa:       {metrics['kappa']:.4f}")
    if metrics["auc_macro"]:
        print(f"  AUC (macro):          {metrics['auc_macro']:.4f}")
        print(f"  AUC (weighted):       {metrics['auc_weighted']:.4f}")
    print(f"  Macro F1:             {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:          {metrics['weighted_f1']:.4f}")
    print(f"  G-mean:               {metrics['g_mean']:.4f}")


# ════════════════════════════════════════════════════════════
# 绘图
# ════════════════════════════════════════════════════════════
def plot_comparison_roc(all_metrics, y_test, filename):
    """绘制多模型ROC曲线对比图"""
    y_arr = y_test.values if hasattr(y_test, "values") else y_test
    y_bin = label_binarize(y_arr, classes=list(range(K)))
    colors_per_model = {
        "ordered_logit": "#E24B4A",
        "ordered_logit_smote": "#BA7517",
        "xgboost": "#378ADD",
        "lightgbm": "#3B6D11",
    }
    model_labels = {
        "ordered_logit": "有序逻辑回归(基线)",
        "ordered_logit_smote": "有序逻辑回归+SMOTE",
        "xgboost": "XGBoost+SMOTE",
        "lightgbm": "LightGBM+SMOTE",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for cls_idx in range(K):
        ax = axes[cls_idx]
        for m in all_metrics:
            proba = m["y_proba"]
            model_type = m["model_type"]
            if y_bin[:, cls_idx].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, cls_idx], proba[:, cls_idx])
            auc_val = sk_auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors_per_model.get(model_type, "#888"),
                    lw=2, label=f"{model_labels.get(model_type, model_type)} (AUC={auc_val:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel("假阳性率 (FPR)")
        ax.set_ylabel("真阳性率 (TPR)")
        ax.set_title(f"ROC - {RISK_LABELS[cls_idx]}", fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("ROC曲线对比 - 4个模型", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(p, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [图表] ROC对比 -> {p}")


def plot_comparison_confusion(all_metrics, filename):
    """绘制多模型混淆矩阵对比图"""
    model_labels = {
        "ordered_logit": "有序逻辑回归(基线)",
        "ordered_logit_smote": "有序逻辑回归+SMOTE",
        "xgboost": "XGBoost+SMOTE",
        "lightgbm": "LightGBM+SMOTE",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
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
        model_type = m["model_type"]
        ax.set_title(f"{model_labels.get(model_type, model_type)}\n(n={total:,})", fontsize=11, fontweight="bold")

    plt.suptitle("混淆矩阵对比", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(p, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [图表] 混淆矩阵对比 -> {p}")


def plot_metrics_bar(summary_df, filename):
    """绘制综合指标柱状图对比"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    metric_names = ["kappa", "auc_macro", "macro_f1", "weighted_f1", "g_mean"]
    metric_labels = ["Cohen's Kappa", "AUC (macro)", "Macro F1", "Weighted F1", "G-mean"]
    bar_colors = ["#E24B4A", "#BA7517", "#378ADD", "#3B6D11"]

    model_names = summary_df["模型"].tolist()

    for idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[idx]
        values = summary_df[metric].tolist()
        bars = ax.bar(range(len(model_names)), values, color=bar_colors, edgecolor="white", width=0.6)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        # 在柱子顶部显示数值
        for bar, val in zip(bars, values):
            if val is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 第6个子图：各类别召回率对比
    ax = axes[5]
    x = np.arange(K)
    width = 0.2
    for i, (_, row) in enumerate(summary_df.iterrows()):
        recalls = [row[f"recall_{g}"] for g in range(K)]
        ax.bar(x + i * width, recalls, width, label=row["模型"], color=bar_colors[i])
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(RISK_LABELS, fontsize=9)
    ax.set_ylabel("Recall", fontsize=11)
    ax.set_title("各类别召回率对比", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("模型性能指标全面对比", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(p, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [图表] 指标对比柱状图 -> {p}")


# ════════════════════════════════════════════════════════════
# 保存评估报告
# ════════════════════════════════════════════════════════════
def save_report(all_metrics, y_test, summary_df):
    """保存完整的评估报告到TXT和CSV"""
    lines = []
    lines.append("=" * 70)
    lines.append("SMOTE优化实验报告 - 策略2完整方案")
    lines.append("=" * 70)
    lines.append(f"\n生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"测试集样本量: {len(y_test):,}")
    lines.append(f"\n优化策略: 保留4类 + SMOTE重采样 + 类别权重 + 树模型")
    lines.append(f"\n对比模型:")
    for i, m in enumerate(all_metrics):
        lines.append(f"  {i+1}. {m['model_name']}")

    # 各模型详细结果
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
            lines.append(f"    各类别AUC:            { {RISK_LABELS[k]: v for k, v in m['per_class_auc'].items()} }")
        lines.append(f"    Macro F1:             {m['macro_f1']:.4f}")
        lines.append(f"    Weighted F1:          {m['weighted_f1']:.4f}")
        lines.append(f"    G-mean:               {m['g_mean']:.4f}")

    # 对比总结
    lines.append(f"\n{'='*70}")
    lines.append(f"  综合对比总结")
    lines.append(f"{'='*70}")
    lines.append(f"\n{summary_df.to_string(index=False)}")

    lines.append(f"\n{'='*70}")
    lines.append(f"  结论与建议")
    lines.append(f"{'='*70}")
    best_kappa = summary_df.loc[summary_df["kappa"].idxmax()]
    best_f1 = summary_df.loc[summary_df["macro_f1"].idxmax()]
    best_auc = summary_df.loc[summary_df["auc_macro"].idxmax()]
    lines.append(f"\n  Kappa最高: {best_kappa['模型']} (Kappa={best_kappa['kappa']})")
    lines.append(f"  Macro F1最高: {best_f1['模型']} (F1={best_f1['macro_f1']})")
    lines.append(f"  AUC最高: {best_auc['模型']} (AUC={best_auc['auc_macro']})")

    report = "\n".join(lines)
    report_path = os.path.join(OUTPUT_DIR, "08_smote_optimization_report.txt")
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
    print("  SMOTE优化实验 - 策略2完整方案")
    print("  保留4类 + SMOTE + 类别权重 + 树模型")
    print("=" * 60)

    # 1. 数据准备
    X_train, X_test, y_train, y_test, scaler, feat_names = load_and_prepare()

    # 2. SMOTE重采样（中等策略：避免过度合成）
    X_train_smote, y_train_smote = apply_smote(X_train, y_train, strategy="moderate")

    # 3. 训练4个模型
    model1, type1 = train_ordered_logit(X_train, y_train)
    model2, type2 = train_ordered_logit_smote(X_train_smote, y_train_smote)
    model3, type3 = train_xgboost(X_train_smote, y_train_smote, y_train)
    model4, type4 = train_lightgbm(X_train_smote, y_train_smote, y_train)

    # 4. 评估（全部在原始不平衡测试集上）
    all_models = [
        (model1, type1, "有序逻辑回归(基线)"),
        (model2, type2, "有序逻辑回归+SMOTE"),
        (model3, type3, "XGBoost+SMOTE"),
        (model4, type4, "LightGBM+SMOTE"),
    ]

    all_metrics = []
    for model, mtype, mname in all_models:
        m = evaluate(model, mtype, X_test, y_test, mname)
        all_metrics.append(m)
        print_evaluation(m, y_test)

    # 5. 构建对比总结表
    summary_rows = []
    model_short_names = {
        "ordered_logit": "有序逻辑回归(基线)",
        "ordered_logit_smote": "有序逻辑回归+SMOTE",
        "xgboost": "XGBoost+SMOTE",
        "lightgbm": "LightGBM+SMOTE",
    }
    for m in all_metrics:
        row = {
            "模型": model_short_names.get(m["model_type"], m["model_name"]),
            "kappa": m["kappa"],
            "auc_macro": m["auc_macro"],
            "auc_weighted": m["auc_weighted"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "g_mean": m["g_mean"],
        }
        for i in range(K):
            row[f"precision_{i}"] = m["per_class_prf"][i]["precision"]
            row[f"recall_{i}"] = m["per_class_prf"][i]["recall"]
            row[f"f1_{i}"] = m["per_class_prf"][i]["f1"]
            row[f"auc_class_{i}"] = m["per_class_auc"].get(i)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # 保存对比CSV
    csv_path = os.path.join(OUTPUT_DIR, "09_model_comparison.csv")
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  [对比表] -> {csv_path}")

    # 6. 绘图
    print(f"\n[生成对比图表]")
    plot_comparison_roc(all_metrics, y_test, "10_roc_comparison.png")
    plot_comparison_confusion(all_metrics, "11_confusion_matrix_comparison.png")
    plot_metrics_bar(summary_df, "12_metrics_comparison.png")

    # 7. 保存完整报告
    save_report(all_metrics, y_test, summary_df)

    # 汇总
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  优化实验完成，耗时 {elapsed:.1f}s")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
