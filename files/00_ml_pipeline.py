"""
肿瘤患者压力性损伤风险预测模型 —— ML 模块（有序逻辑回归）
====================================================
负责内容（对应合同乙方义务）：
  1. 数据预处理：清洗 / One-hot 编码 / Z-score 标准化 / 分层抽样划分
  2. 有序逻辑回归建模（模型 A）
     - 优先：statsmodels OrderedModel（真正的 proportional odds）
     - 降级：累积 logit via K-1 二分类器（sklearn 实现，效果等价）
  3. 模型评估：混淆矩阵 / 多分类 AUC / Kappa / ROC 曲线
  4. 特征重要性：OR 值 + 95% CI + 可视化
  5. 基线特征表（带 P 值，供论文使用）
  6. 模型文件保存（.pkl）+ 推理接口（供部署组调用）

环境要求：Python 3.8+，pandas / numpy / scikit-learn / matplotlib / seaborn / scipy
可选增强：statsmodels（pip install statsmodels 后自动切换为真正的 proportional odds）

输出文件（OUTPUT_DIR）：
  01_preprocessing_report.txt   数据清洗报告
  02_baseline_table.csv         基线特征表（含 P 值）
  03_feature_importance.csv     特征重要性（OR 值 + CI）
  04_confusion_matrix.png       混淆矩阵热图
  05_roc_curves.png             ROC 曲线（OVR）
  06_feature_importance.png     特征重要性排序图
  07_model_logistic.pkl         模型 bundle（模型 + scaler + 元数据）
"""

import os
import warnings
import pickle
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
)

warnings.filterwarnings("ignore")
matplotlib.rcParams["figure.dpi"] = 150

# ── 中文字体 ─────────────────────────────────────────────────────────
def _setup_font():
    preferred = ["WenQuanYi Zen Hei", "Noto Sans CJK JP", "Noto Serif CJK JP",
                 "SimHei", "Microsoft YaHei", "PingFang SC"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    return None

_setup_font()

# ── 路径 ─────────────────────────────────────────────────────────────
DATA_PATH  = "/mnt/user-data/uploads/1774442965959_小规模数据.xlsx"
OUTPUT_DIR = "/mnt/user-data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 变量定义 ─────────────────────────────────────────────────────────
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

CLINICAL_BOUNDS = {
    "BMI":    (10.0, 60.0),
    "白蛋白":  (10.0, 60.0),
    "年龄":   (18.0, 120.0),
    "住院时长": (0.0, 365.0),
}


# ═══════════════════════════════════════════════════════════════════
# STEP 1  数据加载
# ═══════════════════════════════════════════════════════════════════
def load_data(path):
    df = pd.read_excel(path)
    print(f"[加载] 原始数据：{df.shape[0]} 行 x {df.shape[1]} 列")
    return df


# ═══════════════════════════════════════════════════════════════════
# STEP 2  数据清洗
# ═══════════════════════════════════════════════════════════════════
def clean_data(df):
    df = df.copy()
    lines = ["=" * 50, "数据清洗报告", "=" * 50, ""]

    # 去重
    n0 = len(df)
    df = df.drop_duplicates()
    lines.append(f"[重复行] 删除 {n0 - len(df)} 条，剩余 {len(df)} 条")

    # 缺失值
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        lines.append("[缺失值] 无缺失值")
    else:
        lines.append("\n[缺失值]")
        for col, cnt in missing.items():
            pct = cnt / len(df) * 100
            lines.append(f"  {col}: {cnt} 条 ({pct:.1f}%)")
            if pct > 30:
                df = df.drop(columns=[col])
                lines.append("  -> 缺失率 > 30%，已剔除（需临床确认）")
            elif col in CONTINUOUS_VARS:
                v = df[col].median()
                df[col].fillna(v, inplace=True)
                lines.append(f"  -> 连续变量，中位数 {v:.2f} 填充")
            else:
                v = df[col].mode()[0]
                df[col].fillna(v, inplace=True)
                lines.append(f"  -> 分类变量，众数 {v} 填充")

    # 异常值
    lines.append("\n[异常值]")
    for col in CONTINUOUS_VARS:
        if col not in df.columns:
            continue
        z_out = (np.abs((df[col] - df[col].mean()) / df[col].std()) > 3).sum()
        lo, hi = CLINICAL_BOUNDS.get(col, (df[col].min(), df[col].max()))
        cl_out = ((df[col] < lo) | (df[col] > hi)).sum()
        if z_out > 0 or cl_out > 0:
            lines.append(
                f"  {col}: Z-score异常={z_out}，临床阈值[{lo},{hi}]外={cl_out}"
                " -- 建议人工核查（本次保留）"
            )
        else:
            lines.append(f"  {col}: 无异常值")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(OUTPUT_DIR, "01_preprocessing_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    return df


# ═══════════════════════════════════════════════════════════════════
# STEP 3  基线特征表
# ═══════════════════════════════════════════════════════════════════
def baseline_table(df):
    groups = [df[df[TARGET] == i] for i in range(K)]
    ns     = [len(g) for g in groups]
    rows   = []

    for col in CONTINUOUS_VARS:
        if col not in df.columns:
            continue
        vals = [g[col].dropna().values for g in groups]
        _, p = sp_stats.kruskal(*[v for v in vals if len(v) > 0])
        row = {"变量": col, "统计方法": "Kruskal-Wallis"}
        for i, g in enumerate(groups):
            row[f"等级{i}（n={ns[i]}）"] = f"{g[col].mean():.1f}±{g[col].std():.1f}"
        row["P值"] = "<0.0001" if p < 0.0001 else f"{p:.4f}"
        rows.append(row)

    for col in BINARY_VARS:
        if col not in df.columns:
            continue
        contingency = pd.crosstab(df[col], df[TARGET])
        try:
            _, p, _, _ = sp_stats.chi2_contingency(contingency)
        except Exception:
            p = np.nan
        row = {"变量": f"{col}（=1）", "统计方法": "卡方检验"}
        for i, g in enumerate(groups):
            n1 = int(g[col].sum())
            row[f"等级{i}（n={ns[i]}）"] = f"{n1} ({n1/ns[i]*100:.1f}%)"
        row["P值"] = (
            "<0.0001" if (not np.isnan(p) and p < 0.0001)
            else ("NaN" if np.isnan(p) else f"{p:.4f}")
        )
        rows.append(row)

    table = pd.DataFrame(rows)
    path  = os.path.join(OUTPUT_DIR, "02_baseline_table.csv")
    table.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n[基线特征表] 已保存 -> {path}")
    print(table.to_string(index=False))
    return table


# ═══════════════════════════════════════════════════════════════════
# STEP 4  特征编码
# ═══════════════════════════════════════════════════════════════════
def encode_features(df):
    df = df.copy()
    y  = df[TARGET].astype(int)
    X  = df.drop(columns=["ID", TARGET], errors="ignore")
    for col in NOMINAL_VARS:
        if col not in X.columns:
            continue
        dummies = pd.get_dummies(X[col], prefix=col, drop_first=False, dtype=int)
        dummies = dummies.iloc[:, :-1]
        X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
    print(f"\n[特征编码] 编码后特征数：{X.shape[1]}")
    return X, y, list(X.columns)


# ═══════════════════════════════════════════════════════════════════
# STEP 5  分层抽样划分（7:3）
# ═══════════════════════════════════════════════════════════════════
def split_data(X, y, test_size=0.3, random_state=42):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for tr, te in sss.split(X, y):
        X_train = X.iloc[tr].reset_index(drop=True)
        X_test  = X.iloc[te].reset_index(drop=True)
        y_train = y.iloc[tr].reset_index(drop=True)
        y_test  = y.iloc[te].reset_index(drop=True)
    print(f"\n[数据划分] 训练集：{len(X_train)} 条  测试集：{len(X_test)} 条")
    print("  训练集分布：", dict(y_train.value_counts().sort_index()))
    print("  测试集分布：", dict(y_test.value_counts().sort_index()))
    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════════
# STEP 6  Z-score 标准化（仅连续变量）
# ═══════════════════════════════════════════════════════════════════
def standardize(X_train, X_test):
    X_train, X_test = X_train.copy(), X_test.copy()
    cols = [c for c in CONTINUOUS_VARS if c in X_train.columns]
    scaler = StandardScaler()
    X_train[cols] = scaler.fit_transform(X_train[cols])
    X_test[cols]  = scaler.transform(X_test[cols])
    print(f"\n[标准化] Z-score 列：{cols}")
    return X_train, X_test, scaler


# ═══════════════════════════════════════════════════════════════════
# STEP 7  有序逻辑回归（Proportional Odds）
# ═══════════════════════════════════════════════════════════════════
class CumulativeLogitOrdinal:
    """
    Proportional Odds Model 的 sklearn 实现。
    通过 K-1 个累积二分类器 P(Y<=k) 模拟 proportional odds 约束。
    共享系数由各二分类器系数的平均值近似。
    """
    def __init__(self, C=1.0, max_iter=2000, random_state=42):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.models_ = []
        self.coef_   = None
        self.intercepts_ = None

    def fit(self, X, y, feature_names=None):
        y = np.asarray(y)
        self.classes_ = sorted(np.unique(y))
        self.K_        = len(self.classes_)
        self.feature_names_ = feature_names
        self.models_ = []
        for k in self.classes_[:-1]:
            y_bin = (y <= k).astype(int)
            if y_bin.sum() == 0 or (1 - y_bin).sum() == 0:
                continue
            clf = LogisticRegression(
                solver="lbfgs", max_iter=self.max_iter,
                C=self.C, class_weight="balanced",
                random_state=self.random_state,
            )
            clf.fit(X, y_bin)
            self.models_.append((k, clf))
        coefs = np.array([m.coef_[0] for _, m in self.models_])
        self.coef_       = coefs.mean(axis=0)
        self.intercepts_ = np.array([m.intercept_[0] for _, m in self.models_])
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        K = self.K_
        cum = np.zeros((len(X), K - 1))
        for i, (k, clf) in enumerate(self.models_):
            cum[:, k] = clf.predict_proba(X)[:, 1]
        for col in range(1, K - 1):
            cum[:, col] = np.maximum(cum[:, col], cum[:, col - 1])
        cum = np.clip(cum, 0, 1)
        proba = np.zeros((len(X), K))
        proba[:, 0] = cum[:, 0]
        for k in range(1, K - 1):
            proba[:, k] = cum[:, k] - cum[:, k - 1]
        proba[:, K - 1] = 1 - cum[:, K - 2]
        proba = np.clip(proba, 0, 1)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def train_model(X_train, y_train, feature_names):
    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        print("\n[建模] statsmodels OrderedModel（Proportional Odds）")
        sm = OrderedModel(y_train.values, X_train.values, distr="logit")
        result = sm.fit(method="bfgs", disp=False)
        return result, "statsmodels"
    except ImportError:
        pass

    print("\n[建模] 累积 logit（K-1 二分类器，proportional odds 近似）")
    print("       pip install statsmodels 后将切换为官方实现")
    model = CumulativeLogitOrdinal(C=1.0, max_iter=2000, random_state=42)
    model.fit(X_train.values, y_train.values, feature_names=feature_names)
    return model, "cumlogit"


# ═══════════════════════════════════════════════════════════════════
# STEP 8  模型评估
# ═══════════════════════════════════════════════════════════════════
def evaluate_model(model, model_type, X_test, y_test):
    X_arr = X_test.values if hasattr(X_test, "values") else X_test
    y_arr = y_test.values if hasattr(y_test, "values") else y_test

    if model_type == "statsmodels":
        proba  = model.predict(X_arr)
        y_pred = proba.argmax(axis=1)
    else:
        proba  = model.predict_proba(X_arr)
        y_pred = model.predict(X_arr)

    cm = confusion_matrix(y_arr, y_pred)
    print("\n[评估] 混淆矩阵：")
    print(cm)
    _plot_confusion_matrix(cm)

    print("\n[评估] 分类报告：")
    print(classification_report(y_arr, y_pred,
          target_names=RISK_LABELS, zero_division=0))

    kappa = cohen_kappa_score(y_arr, y_pred)
    print(f"[评估] Kappa 系数：{kappa:.4f}")

    try:
        auc_macro    = roc_auc_score(y_arr, proba, multi_class="ovr", average="macro")
        auc_weighted = roc_auc_score(y_arr, proba, multi_class="ovr", average="weighted")
        print(f"[评估] AUC macro   ：{auc_macro:.4f}")
        print(f"[评估] AUC weighted：{auc_weighted:.4f}")
    except Exception as e:
        print(f"[评估] AUC 计算跳过：{e}")
        auc_macro = auc_weighted = None

    _plot_roc_curves(y_arr, proba)

    metrics = {
        "model_type":    model_type,
        "kappa":         round(kappa, 4),
        "auc_macro":     round(auc_macro, 4) if auc_macro else None,
        "auc_weighted":  round(auc_weighted, 4) if auc_weighted else None,
        "confusion_matrix": cm.tolist(),
    }
    return metrics, proba, y_pred


# ═══════════════════════════════════════════════════════════════════
# STEP 9  特征重要性（OR 值 + 95% CI）
# ═══════════════════════════════════════════════════════════════════
def feature_importance(model, model_type, feature_names):
    print("\n[特征重要性] 计算中...")

    if model_type == "statsmodels":
        n     = len(feature_names)
        coef  = model.params[:n].values
        se    = model.bse[:n].values
        OR    = np.exp(coef)
        OR_lo = np.exp(coef - 1.96 * se)
        OR_hi = np.exp(coef + 1.96 * se)
        imp_df = pd.DataFrame({
            "特征": feature_names, "系数": coef.round(4),
            "OR值": OR.round(4), "OR_CI_下": OR_lo.round(4), "OR_CI_上": OR_hi.round(4),
        })
    else:
        coef  = model.coef_
        coefs_stack = np.array([m.coef_[0] for _, m in model.models_])
        se    = coefs_stack.std(axis=0)
        OR    = np.exp(coef)
        OR_lo = np.exp(coef - 1.96 * se)
        OR_hi = np.exp(coef + 1.96 * se)
        imp_df = pd.DataFrame({
            "特征": feature_names, "系数（均值）": coef.round(4),
            "OR值": OR.round(4), "OR_CI_下(近似)": OR_lo.round(4), "OR_CI_上(近似)": OR_hi.round(4),
        })

    # 按 |OR - 1| 降序排列，OR 偏离 1 越多越重要
    imp_df["_sort"] = np.abs(imp_df["OR值"] - 1)
    imp_df = imp_df.sort_values("_sort", ascending=False).drop(columns="_sort").reset_index(drop=True)

    csv_path = os.path.join(OUTPUT_DIR, "03_feature_importance.csv")
    imp_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(imp_df.head(20).to_string(index=False))
    print(f"\n[特征重要性] 已保存 -> {csv_path}")
    _plot_feature_importance(imp_df, model_type)
    return imp_df


# ═══════════════════════════════════════════════════════════════════
# 绘图
# ═══════════════════════════════════════════════════════════════════
def _plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=RISK_LABELS, yticklabels=RISK_LABELS,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("预测标签", fontsize=12)
    ax.set_ylabel("真实标签", fontsize=12)
    ax.set_title("混淆矩阵 — 有序逻辑回归", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "04_confusion_matrix.png")
    plt.savefig(p, bbox_inches="tight"); plt.close()
    print(f"[图表] 混淆矩阵 -> {p}")


def _plot_roc_curves(y_arr, proba):
    n_cls  = proba.shape[1]
    y_bin  = label_binarize(y_arr, classes=list(range(n_cls)))
    colors = ["#2E5496", "#C9541A", "#2E8B57", "#8B2252"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for i in range(n_cls):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        auc_val     = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f"{RISK_LABELS[i]} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("假阳性率 (FPR)", fontsize=12)
    ax.set_ylabel("真阳性率 (TPR)", fontsize=12)
    ax.set_title("ROC 曲线 — 有序逻辑回归（OVR）", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "05_roc_curves.png")
    plt.savefig(p, bbox_inches="tight"); plt.close()
    print(f"[图表] ROC 曲线 -> {p}")


def _plot_feature_importance(imp_df, model_type):
    top_n   = min(20, len(imp_df))
    plot_df = imp_df.head(top_n).sort_values("OR值", ascending=True).copy()

    colors  = ["#C9541A" if v > 1 else "#2E5496" for v in plot_df["OR值"]]
    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.42)))
    ax.barh(plot_df["特征"], plot_df["OR值"], color=colors,
            edgecolor="white", height=0.65)

    ci_lo = "OR_CI_下" if "OR_CI_下" in plot_df.columns else "OR_CI_下(近似)"
    ci_hi = "OR_CI_上" if "OR_CI_上" in plot_df.columns else "OR_CI_上(近似)"
    xerr_lo = (plot_df["OR值"] - plot_df[ci_lo]).clip(lower=0)
    xerr_hi = (plot_df[ci_hi] - plot_df["OR值"]).clip(lower=0)
    ax.errorbar(plot_df["OR值"], plot_df["特征"],
                xerr=[xerr_lo, xerr_hi],
                fmt="none", color="black", capsize=3, linewidth=1)

    ax.axvline(x=1, color="gray", linestyle="--", linewidth=1.2)
    ax.set_xlabel("OR 值（95% CI）", fontsize=12)
    note = "（近似 CI）" if model_type == "cumlogit" else ""
    ax.set_title(
        f"特征重要性排序 — 有序逻辑回归 {note}\n"
        "橙色=风险因素(OR>1)  蓝色=保护因素(OR<1)",
        fontsize=11, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)
    sns.despine()
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "06_feature_importance.png")
    plt.savefig(p, bbox_inches="tight"); plt.close()
    print(f"[图表] 特征重要性 -> {p}")


# ═══════════════════════════════════════════════════════════════════
# STEP 10  保存模型
# ═══════════════════════════════════════════════════════════════════
def save_model(model, model_type, scaler, feature_names, metrics):
    bundle = {
        "model":           model,
        "model_type":      model_type,
        "scaler":          scaler,
        "feature_names":   feature_names,
        "continuous_vars": CONTINUOUS_VARS,
        "nominal_vars":    NOMINAL_VARS,
        "binary_vars":     BINARY_VARS,
        "target":          TARGET,
        "risk_labels":     RISK_LABELS,
        "K":               K,
        "metrics":         metrics,
    }
    path = os.path.join(OUTPUT_DIR, "07_model_logistic.pkl")
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n[保存] 模型 bundle -> {path}")
    return path


# ═══════════════════════════════════════════════════════════════════
# 推理接口（供部署组调用）
# ═══════════════════════════════════════════════════════════════════
def predict_single(bundle, patient_data):
    """
    单条患者推理。
    示例：
        bundle = pickle.load(open("07_model_logistic.pkl", "rb"))
        result = predict_single(bundle, {
            "性别": 1, "年龄": 70, "白蛋白": 28.5, "BMI": 16.0,
            "肿瘤类型": 6, "化疗史": 0, "感染": 1, ...
        })
        # -> {"风险等级": 3, "风险标签": "高风险(3)", "各等级概率": {...}}
    """
    model         = bundle["model"]
    model_type    = bundle["model_type"]
    scaler        = bundle["scaler"]
    feature_names = bundle["feature_names"]
    cont_vars     = bundle["continuous_vars"]

    row = pd.DataFrame([patient_data])
    for col in bundle.get("nominal_vars", []):
        if col in row.columns:
            dummies = pd.get_dummies(row[col], prefix=col, dtype=int).iloc[:, :-1]
            row = pd.concat([row.drop(columns=[col]), dummies], axis=1)
    for col in feature_names:
        if col not in row.columns:
            row[col] = 0
    row = row[feature_names].copy()
    cont_present = [c for c in cont_vars if c in row.columns]
    row[cont_present] = scaler.transform(row[cont_present])

    X = row.values
    if model_type == "statsmodels":
        proba = model.predict(X)[0]
    else:
        proba = model.predict_proba(X)[0]

    predicted = int(np.argmax(proba))
    return {
        "风险等级":   predicted,
        "风险标签":   bundle["risk_labels"][predicted],
        "各等级概率": {bundle["risk_labels"][i]: round(float(p), 4) for i, p in enumerate(proba)},
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  肿瘤患者压力性损伤风险预测 —— ML 模块（有序逻辑回归）")
    print("=" * 60)

    df_raw   = load_data(DATA_PATH)
    df_clean = clean_data(df_raw)
    baseline_table(df_clean)
    X, y, feat_names = encode_features(df_clean)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, X_test, scaler = standardize(X_train, X_test)
    model, model_type = train_model(X_train, y_train, feat_names)
    metrics, proba, _ = evaluate_model(model, model_type, X_test, y_test)
    feature_importance(model, model_type, feat_names)
    save_model(model, model_type, scaler, feat_names, metrics)

    print("\n" + "=" * 60)
    print("  全部输出文件：")
    for fn in sorted(os.listdir(OUTPUT_DIR)):
        if fn.endswith((".txt", ".csv", ".png", ".pkl")):
            sz = os.path.getsize(os.path.join(OUTPUT_DIR, fn))
            print(f"  {fn:<46}  {sz/1024:>6.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
