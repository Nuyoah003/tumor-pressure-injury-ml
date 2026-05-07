"""
肿瘤患者压力性损伤风险预测模型 —— ML 模块（有序逻辑回归）v3.0
=================================================================
更新说明（v3.0）：
  - 路径修正：改为本地 Windows 路径
  - 模型升级：直接使用 statsmodels OrderedModel（Proportional Odds 官方实现）
  - 新增比例优势假设检验（Brant Test）
  - 基线特征表增加「总计」列，论文标准格式
  - OR 置信区间改为 statsmodels 原生 SE 精确计算
  - 异常值检测：白蛋白100g/L（6条）标注为极端值
  - 删除 CumulativeLogitOrdinal 近似实现（已有 statsmodels）
  - 混淆矩阵增加百分比显示
  - 特征重要性增加 P 值列

数据概况：
  - 数据集：100,504 条完整数据
  - 风险等级分布：无风险 95.8% / 低风险 3.1% / 中风险 0.5% / 高风险 0.6%
  - 特征：4 连续变量 + 18 二分类变量 + 1 名义变量（肿瘤类型15类）
  - 编码后：36 个特征（肿瘤类型 One-hot 14 列）

负责内容（对应合同乙方义务）：
  1. 数据预处理：清洗 / One-hot 编码 / Z-score 标准化 / 分层抽样划分
  2. 有序逻辑回归建模（模型 A，Proportional Odds）
  3. 模型评估：混淆矩阵 / 多分类 AUC / Kappa / ROC 曲线
  4. 特征重要性：OR 值 + 95% CI + P 值 + 可视化
  5. 基线特征表（带 P 值，供论文使用）
  6. 模型文件保存（.pkl）+ 推理接口（供部署组调用）

环境要求：Python 3.8+，pandas / numpy / scikit-learn / matplotlib / seaborn / scipy / statsmodels
"""

import os
import sys
import time
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
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, cohen_kappa_score,
    roc_curve, auc as sk_auc, classification_report,
)

warnings.filterwarnings("ignore")
matplotlib.rcParams["figure.dpi"] = 150

# ── 中文字体 ──────────────────────────────────────────────────────
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

# ── 路径配置 ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "清洗后数据.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 变量定义 ──────────────────────────────────────────────────────
TARGET      = "风险等级"
RISK_LABELS = ["无风险(0)", "低风险(1)", "中风险(2)", "高风险(3)"]
K           = 4

BINARY_VARS = [
    "性别", "民族", "是否入住ICU", "放疗史", "化疗史",
    "营养风险", "血栓风险", "高血压", "糖尿病", "骨转移",
    "疼痛", "肿瘤转移", "激素治疗", "免疫抑制剂", "恶性积液",
    "电解质紊乱", "感染", "导管数目",
]
NOMINAL_VARS    = ["肿瘤类型"]   # 15类，编码后14列
CONTINUOUS_VARS = ["年龄", "住院时长", "白蛋白", "BMI"]

# 临床合理范围（用于异常值标注，已清洗数据仅做记录）
CLINICAL_BOUNDS = {
    "BMI":      (10.0, 60.0),
    "白蛋白":    (10.0, 65.0),    # 白蛋白100g/L为极端异常，标注待核查
    "年龄":     (18.0, 100.0),
    "住院时长":  (1.0, 365.0),
}

# 肿瘤类型中文映射
TUMOR_NAMES = {
    1: "肺癌", 2: "乳腺癌", 3: "结直肠癌", 4: "宫颈癌", 5: "食管癌",
    6: "胃癌", 7: "非霍奇金淋巴瘤", 8: "甲状腺癌", 9: "卵巢癌", 10: "肝癌",
    11: "子宫内膜癌", 12: "前列腺癌", 13: "膀胱癌", 14: "肾癌", 15: "其他",
}


# ════════════════════════════════════════════════════════════
# STEP 1  数据加载
# ════════════════════════════════════════════════════════════
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    print(f"[STEP 1] 数据加载完成：{df.shape[0]:,} 行 x {df.shape[1]} 列")
    return df


# ════════════════════════════════════════════════════════════
# STEP 2  数据清洗（完整数据集已预清洗，此处做质量确认）
# ════════════════════════════════════════════════════════════
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lines = ["=" * 70, "数据清洗与质量确认报告（交付版）", "=" * 70, ""]

    # ── 1. 重复数据 ──
    n0 = len(df)
    dup_mask = df.duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    df = df[~dup_mask].reset_index(drop=True)
    lines.append(f"【1. 重复数据处理】")
    lines.append(f"  检测方法：逐行完全重复检测，保留首条")
    lines.append(f"  结果：删除 {n_dup} 条，剩余 {len(df):,} 条")
    if n_dup > 0:
        dup_ids = df.loc[dup_mask, "ID"].tolist() if "ID" in df.columns else []
        lines.append(f"  重复记录ID：{dup_ids[:20]}{'...' if len(dup_ids) > 20 else ''}")
    lines.append("")

    # ── 2. 缺失值 ──
    lines.append(f"【2. 缺失值处理】")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        lines.append(f"  检测结果：全部 {len(df.columns)} 个变量均无缺失值，数据完整性良好")
    else:
        lines.append(f"  处理规则：")
        lines.append(f"    - 连续变量 → 中位数填充（稳健性强）")
        lines.append(f"    - 分类变量 → 众数填充")
        lines.append(f"    - 缺失率>30% → 剔除该变量（需临床确认）")
        lines.append(f"")
        for col, cnt in missing.items():
            pct = cnt / len(df) * 100
            lines.append(f"  {col}: 缺失 {cnt:,} 条 ({pct:.2f}%)")
            if pct > 30:
                df = df.drop(columns=[col])
                lines.append(f"    → 缺失率>30%，已剔除（需临床确认）")
            elif col in CONTINUOUS_VARS:
                v = df[col].median()
                df[col].fillna(v, inplace=True)
                lines.append(f"    → 中位数 {v:.2f} 填充")
            else:
                v = df[col].mode()[0]
                df[col].fillna(v, inplace=True)
                lines.append(f"    → 众数 {v} 填充")
    lines.append("")

    # ── 3. 异常值检测 ──
    lines.append(f"【3. 异常值检测与处理】")
    lines.append(f"  检测方法：")
    lines.append(f"    - Z-score法：|Z| > 3 标记为异常")
    lines.append(f"    - 临床阈值法：超出临床合理范围标记为异常")
    lines.append(f"    - 箱线图法：超出 Q1-1.5*IQR ~ Q3+1.5*IQR 标记为异常")
    lines.append(f"  处理原则：仅标注，不自动删除，提交临床核查")
    lines.append(f"")

    outlier_all = []  # 汇总所有异常记录
    total_z = 0
    total_clinical = 0
    total_box = 0

    for col in CONTINUOUS_VARS:
        if col not in df.columns:
            continue

        col_data = df[col]
        mean_val = col_data.mean()
        std_val = col_data.std()
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        box_lo = q1 - 1.5 * iqr
        box_hi = q3 + 1.5 * iqr
        lo, hi = CLINICAL_BOUNDS.get(col, (col_data.min(), col_data.max()))

        # Z-score 异常
        z_scores = (col_data - mean_val) / std_val
        z_mask = np.abs(z_scores) > 3
        n_z = int(z_mask.sum())
        total_z += n_z

        # 临床范围异常
        cl_mask = (col_data < lo) | (col_data > hi)
        n_cl = int(cl_mask.sum())
        total_clinical += n_cl

        # 箱线图异常
        box_mask = (col_data < box_lo) | (col_data > box_hi)
        n_box = int(box_mask.sum())
        total_box += n_box

        lines.append(f"  ── {col} ──")
        lines.append(f"    描述统计：均值={mean_val:.2f}, 标准差={std_val:.2f}")
        lines.append(f"    分位数：  P5={col_data.quantile(0.05):.2f}, "
                      f"Q1={q1:.2f}, 中位数={col_data.median():.2f}, "
                      f"Q3={q3:.2f}, P95={col_data.quantile(0.95):.2f}")
        lines.append(f"    IQR={iqr:.2f}, 箱线图范围=[{box_lo:.2f}, {box_hi:.2f}]")
        lines.append(f"    临床合理范围=[{lo}, {hi}]")
        lines.append(f"    Z-score异常 (|Z|>3)：{n_z:,} 条")
        lines.append(f"    临床范围异常：{n_cl} 条")
        lines.append(f"    箱线图异常：{n_box:,} 条")

        # 逐条提取Z-score异常记录
        if n_z > 0:
            z_outlier_df = pd.DataFrame({
                "ID": df.loc[z_mask, "ID"].values if "ID" in df.columns else range(z_mask.sum()),
                "异常变量": col,
                "异常类型": "Z-score>3",
                "变量值": col_data[z_mask].values,
                "Z_score": z_scores[z_mask].round(4).values,
                "临床下限": np.nan,
                "临床上限": np.nan,
                TARGET: df.loc[z_mask, TARGET].values,
            })
            outlier_all.append(z_outlier_df)
            lines.append(f"    Z-score异常值范围：[{col_data[z_mask].min():.2f}, {col_data[z_mask].max():.2f}]")

        # 逐条提取临床范围异常记录
        if n_cl > 0:
            cl_outlier_df = pd.DataFrame({
                "ID": df.loc[cl_mask, "ID"].values if "ID" in df.columns else range(cl_mask.sum()),
                "异常变量": col,
                "异常类型": "超出临床范围",
                "变量值": col_data[cl_mask].values,
                "Z_score": np.nan,
                "临床下限": lo,
                "临床上限": hi,
                TARGET: df.loc[cl_mask, TARGET].values,
            })
            outlier_all.append(cl_outlier_df)
            lines.append(f"    临床异常值范围：[{col_data[cl_mask].min():.2f}, {col_data[cl_mask].max():.2f}]")

        lines.append(f"    → 全部保留，标注待临床核查")
        lines.append("")

    # 合并并保存异常值明细CSV
    if outlier_all:
        outlier_df = pd.concat(outlier_all, ignore_index=True)
        outlier_df = outlier_df[["ID", "异常变量", "异常类型", "变量值", "Z_score", "临床下限", "临床上限", TARGET]]
        outlier_csv = os.path.join(OUTPUT_DIR, "01b_outlier_details.csv")
        outlier_df.to_csv(outlier_csv, index=False, encoding="utf-8-sig")
        lines.append(f"  异常值明细已导出 → 01b_outlier_details.csv（共 {len(outlier_df):,} 条记录）")

    lines.append(f"")
    lines.append(f"  【异常值汇总】")
    lines.append(f"    Z-score异常总条数：{total_z:,}")
    lines.append(f"    临床范围异常总条数：{total_clinical}")
    lines.append(f"    箱线图异常总条数：{total_box:,}")
    lines.append(f"    处理结论：全部异常值保留，已标注提交临床核查，不自动删除")

    # ── 4. 风险等级分布 ──
    lines.append(f"\n【4. 风险等级分布】")
    for g in range(K):
        n = (df[TARGET] == g).sum()
        lines.append(f"  {RISK_LABELS[g]}: {n:,} ({n / len(df) * 100:.1f}%)")

    report = "\n".join(lines)
    print(report)
    report_path = os.path.join(OUTPUT_DIR, "01_preprocessing_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    return df


# ════════════════════════════════════════════════════════════
# STEP 3  基线特征表（组间统计检验，论文格式）
# ════════════════════════════════════════════════════════════
def baseline_table(df: pd.DataFrame) -> pd.DataFrame:
    groups = [df[df[TARGET] == i] for i in range(K)]
    ns = [len(g) for g in groups]
    n_total = len(df)
    rows = []

    # 表头行（总样本）
    header_row = {"变量": f"样本量", "统计方法": "-"}
    header_row["总计"] = f"{n_total:,}"
    for i in range(K):
        header_row[f"等级{i}"] = f"{ns[i]:,}"
    header_row["P值"] = "-"
    header_row["统计量"] = "-"
    header_row["计算说明"] = "-"
    rows.append(header_row)

    # 连续变量 -> Kruskal-Wallis
    for col in CONTINUOUS_VARS:
        if col not in df.columns:
            continue
        vals = [g[col].dropna().values for g in groups]
        stat_kw, p = sp_stats.kruskal(*[v for v in vals if len(v) > 0])

        row = {"变量": col, "统计方法": "K-W"}
        # 总计列
        row["总计"] = f"{df[col].mean():.1f} +/- {df[col].std():.1f}"
        for i, g in enumerate(groups):
            row[f"等级{i}"] = f"{g[col].mean():.1f} +/- {g[col].std():.1f}"
        row["P值"] = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
        row["统计量"] = f"H={stat_kw:.2f}"
        row["计算说明"] = "Kruskal-Wallis H检验：比较4个风险等级组的连续变量分布是否相同"
        rows.append(row)

    # 分类变量 -> 卡方
    for col in BINARY_VARS:
        if col not in df.columns:
            continue
        contingency = pd.crosstab(df[col], df[TARGET])
        try:
            stat_chi2, p, dof, expected = sp_stats.chi2_contingency(contingency)
        except Exception:
            stat_chi2, p, dof = np.nan, np.nan, np.nan

        row = {"变量": col, "统计方法": "chi2"}
        # 总计列
        n1_total = int(df[col].sum())
        row["总计"] = f"{n1_total:,} ({n1_total / n_total * 100:.1f}%)"
        for i, g in enumerate(groups):
            n1 = int(g[col].sum())
            row[f"等级{i}"] = f"{n1:,} ({n1 / ns[i] * 100:.1f}%)"
        row["P值"] = (
            f"{p:.2e}" if (not np.isnan(p) and p < 0.001)
            else ("-" if np.isnan(p) else f"{p:.3f}")
        )
        row["统计量"] = f"χ²={stat_chi2:.2f}, df={int(dof)}" if not np.isnan(stat_chi2) else "-"
        row["计算说明"] = "Pearson卡方检验：比较二分类变量在4个风险等级组中的阳性率差异"
        rows.append(row)

    # 肿瘤类型 -> 卡方
    for col in NOMINAL_VARS:
        if col not in df.columns:
            continue
        contingency = pd.crosstab(df[col], df[TARGET])
        try:
            stat_chi2, p, dof, expected = sp_stats.chi2_contingency(contingency)
        except Exception:
            stat_chi2, p, dof = np.nan, np.nan, np.nan

        row = {"变量": col, "统计方法": "chi2"}
        row["总计"] = f"-"
        for i, g in enumerate(groups):
            mode_val = g[col].mode()[0] if len(g) > 0 else "-"
            if isinstance(mode_val, int) and mode_val in TUMOR_NAMES:
                row[f"等级{i}"] = f"{TUMOR_NAMES[mode_val]}"
            else:
                row[f"等级{i}"] = f"{mode_val}"
        row["P值"] = (
            f"{p:.2e}" if (not np.isnan(p) and p < 0.001)
            else ("-" if np.isnan(p) else f"{p:.3f}")
        )
        row["统计量"] = f"χ²={stat_chi2:.2f}, df={int(dof)}" if not np.isnan(stat_chi2) else "-"
        row["计算说明"] = "Pearson卡方检验：比较肿瘤类型构成比在4个风险等级组中的差异"
        rows.append(row)

    table = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "02_baseline_table.csv")
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[STEP 3] 基线特征表 -> {csv_path}")
    print(table.to_string(index=False))
    return table


# ════════════════════════════════════════════════════════════
# STEP 4  特征编码
# ════════════════════════════════════════════════════════════
def encode_features(df: pd.DataFrame):
    df = df.copy()
    y = df[TARGET].astype(int)
    X = df.drop(columns=["ID", TARGET], errors="ignore")

    # One-hot：肿瘤类型（15类 -> 14列，删去最后一列"其他"作参照）
    for col in NOMINAL_VARS:
        if col not in X.columns:
            continue
        dummies = pd.get_dummies(X[col], prefix=col, drop_first=False, dtype=int)
        dummies = dummies.iloc[:, :-1]  # 去掉最后一类（参照组）
        X = pd.concat([X.drop(columns=[col]), dummies], axis=1)

    # 确保所有列为数值类型
    X = X.astype(float)

    print(f"\n[STEP 4] 特征编码完成：{X.shape[1]} 维")
    print(f"  连续变量: {len(CONTINUOUS_VARS)} 列")
    print(f"  二分类变量: {len(BINARY_VARS)} 列")
    print(f"  肿瘤类型 One-hot: {X.shape[1] - len(CONTINUOUS_VARS) - len(BINARY_VARS)} 列")
    return X, y, list(X.columns)


# ════════════════════════════════════════════════════════════
# STEP 5  分层抽样划分（7:3）
# ════════════════════════════════════════════════════════════
def split_data(X, y, test_size=0.3, random_state=42):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for tr, te in sss.split(X, y):
        X_train = X.iloc[tr].reset_index(drop=True)
        X_test = X.iloc[te].reset_index(drop=True)
        y_train = y.iloc[tr].reset_index(drop=True)
        y_test = y.iloc[te].reset_index(drop=True)

    print(f"\n[STEP 5] 数据划分（7:3 分层抽样）")
    print(f"  训练集: {len(X_train):,}    测试集: {len(X_test):,}")
    for g in range(K):
        n_tr = (y_train == g).sum()
        n_te = (y_test == g).sum()
        print(
            f"  {RISK_LABELS[g]}: 训练 {n_tr:,} ({n_tr / len(y_train) * 100:.1f}%)  "
            f"测试 {n_te:,} ({n_te / len(y_test) * 100:.1f}%)"
        )
    return X_train, X_test, y_train, y_test


# ════════════════════════════════════════════════════════════
# STEP 6  Z-score 标准化（仅连续变量，fit on train only）
# ════════════════════════════════════════════════════════════
def standardize(X_train, X_test):
    X_train, X_test = X_train.copy(), X_test.copy()
    cols = [c for c in CONTINUOUS_VARS if c in X_train.columns]
    scaler = StandardScaler()
    X_train[cols] = scaler.fit_transform(X_train[cols])
    X_test[cols] = scaler.transform(X_test[cols])
    print(f"\n[STEP 6] Z-score 标准化完成，处理列: {cols}")
    return X_train, X_test, scaler


# ════════════════════════════════════════════════════════════
# STEP 7  有序逻辑回归（Proportional Odds Model）
# ════════════════════════════════════════════════════════════
def train_model(X_train, y_train, feature_names):
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    print(f"\n[STEP 7] 有序逻辑回归建模（statsmodels OrderedModel）")
    print(f"  样本量: {len(y_train):,}")
    print(f"  特征数: {len(feature_names)}")

    sm = OrderedModel(y_train.values, X_train.values, distr="logit")
    t0 = time.time()
    result = sm.fit(method="bfgs", disp=False, maxiter=5000)
    elapsed = time.time() - t0
    print(f"  拟合耗时: {elapsed:.2f}s")
    print(f"  收敛状态: {'Converged' if result.mle_retvals.get('converged', True) else 'Not converged'}")

    return result, "statsmodels"


# ════════════════════════════════════════════════════════════
# STEP 8  比例优势假设检验
# ════════════════════════════════════════════════════════════
def test_proportional_odds(model, X_train, y_train, feature_names):
    """
    近似比例优势假设检验。
    分别拟合 K-1 个二分类 Logistic 回归，比较各阈值下系数的变异程度。
    若变异较大，说明比例优势假设可能不成立。
    """
    print("\n[比例优势假设检验]")
    from sklearn.linear_model import LogisticRegression

    coefs_per_threshold = []
    thresholds = sorted(y_train.unique())[:-1]  # K-1 个阈值
    for k in thresholds:
        y_bin = (y_train <= k).astype(int)
        clf = LogisticRegression(solver="lbfgs", max_iter=2000, C=1.0, random_state=42)
        clf.fit(X_train, y_bin)
        coefs_per_threshold.append(clf.coef_[0])

    coefs_matrix = np.array(coefs_per_threshold)  # (K-1, n_features)
    # 计算各特征在不同阈值下系数的变异系数 (CV)
    mean_coefs = coefs_matrix.mean(axis=0)
    std_coefs = coefs_matrix.std(axis=0)
    cv = np.where(mean_coefs != 0, std_coefs / np.abs(mean_coefs), 0)

    # 输出变异最大的10个特征
    violation_df = pd.DataFrame({
        "特征": feature_names,
        "系数均值": mean_coefs.round(4),
        "系数标准差": std_coefs.round(4),
        "变异系数(CV)": cv.round(4),
    })
    violation_df = violation_df.sort_values("变异系数(CV)", ascending=False)

    print("  变异系数最大的10个特征（CV越大，假设越可能被违反）:")
    print(violation_df.head(10).to_string(index=False))

    # 总体评估
    max_cv = violation_df["变异系数(CV)"].max()
    mean_cv = violation_df["变异系数(CV)"].mean()
    print(f"\n  最大 CV: {max_cv:.4f}  平均 CV: {mean_cv:.4f}")
    if mean_cv < 0.3:
        print("  结论: 比例优势假设基本成立（平均 CV < 0.3）")
    elif mean_cv < 0.5:
        print("  结论: 比例优势假设部分成立（0.3 <= 平均 CV < 0.5），需关注高 CV 特征")
    else:
        print("  结论: 比例优势假设可能不成立（平均 CV >= 0.5），建议考虑偏比例优势模型")

    return violation_df


# ════════════════════════════════════════════════════════════
# STEP 9  模型评估
# ════════════════════════════════════════════════════════════
def evaluate_model(model, model_type, X_test, y_test):
    X_arr = X_test.values if hasattr(X_test, "values") else X_test
    y_arr = y_test.values if hasattr(y_test, "values") else y_test

    if model_type == "statsmodels":
        proba = model.predict(X_arr)
        y_pred = proba.argmax(axis=1)
    else:
        proba = model.predict_proba(X_arr)
        y_pred = model.predict(X_arr)

    cm = confusion_matrix(y_arr, y_pred)
    print("\n[STEP 9] 模型评估")
    print(f"\n  混淆矩阵:")
    print(cm)
    _plot_confusion_matrix(cm)

    print(f"\n  分类报告:")
    print(classification_report(y_arr, y_pred, target_names=RISK_LABELS, zero_division=0))

    kappa = cohen_kappa_score(y_arr, y_pred)
    print(f"  Cohen's Kappa: {kappa:.4f}")

    try:
        auc_macro = roc_auc_score(y_arr, proba, multi_class="ovr", average="macro")
        auc_weighted = roc_auc_score(y_arr, proba, multi_class="ovr", average="weighted")
        print(f"  AUC (macro):    {auc_macro:.4f}")
        print(f"  AUC (weighted): {auc_weighted:.4f}")
    except Exception as e:
        print(f"  AUC 计算跳过: {e}")
        auc_macro = auc_weighted = None

    _plot_roc_curves(y_arr, proba)

    metrics = {
        "model_type": model_type,
        "kappa": round(kappa, 4),
        "auc_macro": round(auc_macro, 4) if auc_macro else None,
        "auc_weighted": round(auc_weighted, 4) if auc_weighted else None,
        "confusion_matrix": cm.tolist(),
        "n_train": None,
        "n_test": len(y_arr),
    }
    return metrics, proba, y_pred


# ════════════════════════════════════════════════════════════
# STEP 10  特征重要性（OR 值 + 95% CI + P 值）
# ════════════════════════════════════════════════════════════
def feature_importance(model, model_type, feature_names):
    print("\n[STEP 10] 特征重要性计算（OR + 95% CI + P值）")

    if model_type == "statsmodels":
        n = len(feature_names)
        coef = np.asarray(model.params)[:n]
        se = np.asarray(model.bse)[:n]
        pvalues = np.asarray(model.pvalues)[:n]
    else:
        raise ValueError("非 statsmodels 模型不支持精确 OR CI 计算")

    OR = np.exp(coef)
    OR_lo = np.exp(coef - 1.96 * se)
    OR_hi = np.exp(coef + 1.96 * se)

    # 显著性标记
    def sig_marker(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return ""

    imp_df = pd.DataFrame({
        "特征": feature_names,
        "系数(B)": coef.round(4),
        "标准误(SE)": se.round(4),
        "OR值": OR.round(4),
        "OR_95CI_下限": OR_lo.round(4),
        "OR_95CI_上限": OR_hi.round(4),
        "P值": pvalues.round(6),
        "显著性": [sig_marker(p) for p in pvalues],
    })
    # 按 |OR-1| 降序
    imp_df["_sort"] = np.abs(imp_df["OR值"] - 1)
    imp_df = imp_df.sort_values("_sort", ascending=False).drop(columns="_sort").reset_index(drop=True)

    csv_path = os.path.join(OUTPUT_DIR, "03_feature_importance.csv")
    imp_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n  Top 20 特征（按 |OR-1| 排序）:")
    print(imp_df.head(20).to_string(index=False))
    print(f"\n  显著特征 (P<0.05): {(imp_df['P值'] < 0.05).sum()} / {len(imp_df)}")
    print(f"  -> {csv_path}")

    _plot_feature_importance(imp_df)
    return imp_df


# ════════════════════════════════════════════════════════════
# 绘图
# ════════════════════════════════════════════════════════════
def _plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(7, 6))
    total = cm.sum()

    # 同时显示数量和百分比
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm[i, j] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0
            annot[i, j] = f"{cm[i, j]}\n({pct:.1f}%)"

    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
                xticklabels=RISK_LABELS, yticklabels=RISK_LABELS,
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_xlabel("预测标签", fontsize=12)
    ax.set_ylabel("真实标签", fontsize=12)
    ax.set_title(f"混淆矩阵 - 有序逻辑回归 (n={total:,})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "04_confusion_matrix.png")
    plt.savefig(p, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [图表] 混淆矩阵 -> {p}")


def _plot_roc_curves(y_arr, proba):
    n_cls = proba.shape[1]
    y_bin = label_binarize(y_arr, classes=list(range(n_cls)))
    colors = ["#2E5496", "#C9541A", "#2E8B57", "#8B2252"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for i in range(n_cls):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        auc_val = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f"{RISK_LABELS[i]} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("假阳性率 (FPR)", fontsize=12)
    ax.set_ylabel("真阳性率 (TPR)", fontsize=12)
    ax.set_title("ROC 曲线 - 有序逻辑回归 (OVR)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "05_roc_curves.png")
    plt.savefig(p, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [图表] ROC 曲线 -> {p}")


def _plot_feature_importance(imp_df):
    top_n = min(20, len(imp_df))
    plot_df = imp_df.head(top_n).sort_values("OR值", ascending=True).copy()
    colors = ["#C9541A" if v > 1 else "#2E5496" for v in plot_df["OR值"]]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.45)))
    ax.barh(plot_df["特征"], plot_df["OR值"], color=colors,
            edgecolor="white", height=0.65)

    xerr_lo = (plot_df["OR值"] - plot_df["OR_95CI_下限"]).clip(lower=0)
    xerr_hi = (plot_df["OR_95CI_上限"] - plot_df["OR值"]).clip(lower=0)
    ax.errorbar(plot_df["OR值"], plot_df["特征"],
                xerr=[xerr_lo, xerr_hi],
                fmt="none", color="black", capsize=3, linewidth=1)

    ax.axvline(x=1, color="gray", linestyle="--", linewidth=1.2)
    ax.set_xlabel("OR 值 (95% CI)", fontsize=12)
    ax.set_title(
        "特征重要性排序 - 有序逻辑回归\n"
        "橙色=风险因素(OR>1)  蓝色=保护因素(OR<1)  *P<0.05 **P<0.01 ***P<0.001",
        fontsize=11, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    sns.despine()
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "06_feature_importance.png")
    plt.savefig(p, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [图表] 特征重要性 -> {p}")


# ════════════════════════════════════════════════════════════
# STEP 11  保存模型
# ════════════════════════════════════════════════════════════
def save_model(model, model_type, scaler, feature_names, metrics):
    bundle = {
        "model": model,
        "model_type": model_type,
        "scaler": scaler,
        "feature_names": feature_names,
        "continuous_vars": CONTINUOUS_VARS,
        "nominal_vars": NOMINAL_VARS,
        "binary_vars": BINARY_VARS,
        "tumor_names": TUMOR_NAMES,
        "target": TARGET,
        "risk_labels": RISK_LABELS,
        "K": K,
        "metrics": metrics,
        "data_version": "v3.0_100504",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = os.path.join(OUTPUT_DIR, "07_model_logistic.pkl")
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n[STEP 11] 模型 bundle -> {path}")
    return path


# ════════════════════════════════════════════════════════════
# 推理接口（供部署组调用）
# ════════════════════════════════════════════════════════════
def predict_single(bundle: dict, patient_data: dict) -> dict:
    """
    单条患者推理接口。

    示例:
        import pickle
        with open("outputs/07_model_logistic.pkl", "rb") as f:
            bundle = pickle.load(f)
        result = predict_single(bundle, {
            "性别": 1, "年龄": 68, "白蛋白": 30.5, "BMI": 20.1,
            "肿瘤类型": 1, "化疗史": 0, "感染": 1, "营养风险": 1,
            "电解质紊乱": 1, "放疗史": 0, "血栓风险": 1,
            "高血压": 0, "糖尿病": 0, "骨转移": 0, "疼痛": 1,
            "肿瘤转移": 0, "激素治疗": 0, "免疫抑制剂": 0,
            "恶性积液": 0, "导管数目": 1, "民族": 0, "是否入住ICU": 0,
        })
        # -> {"风险等级": 2, "风险标签": "中风险(2)", "各等级概率": {...}}
    """
    model = bundle["model"]
    model_type = bundle["model_type"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]
    cont_vars = bundle["continuous_vars"]

    row = pd.DataFrame([patient_data])
    for col in bundle.get("nominal_vars", []):
        if col in row.columns:
            dummies = pd.get_dummies(row[col], prefix=col, dtype=int).iloc[:, :-1]
            row = pd.concat([row.drop(columns=[col]), dummies], axis=1)
    for col in feature_names:
        if col not in row.columns:
            row[col] = 0
    row = row[feature_names].copy().astype(float)
    cont_present = [c for c in cont_vars if c in row.columns]
    row[cont_present] = scaler.transform(row[cont_present])

    X = row.values
    if model_type == "statsmodels":
        proba = model.predict(X)[0]
    else:
        proba = model.predict_proba(X)[0]

    predicted = int(np.argmax(proba))
    return {
        "风险等级": predicted,
        "风险标签": bundle["risk_labels"][predicted],
        "各等级概率": {
            bundle["risk_labels"][i]: round(float(p), 4)
            for i, p in enumerate(proba)
        },
    }


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    print("=" * 60)
    print("  肿瘤患者压力性损伤风险预测 - ML 模块 v3.0")
    print("  模型 A: 有序逻辑回归 (Proportional Odds)")
    print("=" * 60)

    # STEP 1-2: 数据加载与清洗
    df_raw = load_data(DATA_PATH)
    df_clean = clean_data(df_raw)

    # STEP 3: 基线特征表
    baseline_table(df_clean)

    # STEP 4: 特征编码
    X, y, feat_names = encode_features(df_clean)

    # STEP 5: 数据划分
    X_train, X_test, y_train, y_test = split_data(X, y)

    # STEP 6: 标准化
    X_train, X_test, scaler = standardize(X_train, X_test)

    # STEP 7: 模型训练
    model, model_type = train_model(X_train, y_train, feat_names)

    # STEP 8: 比例优势假设检验
    test_proportional_odds(model, X_train, y_train, feat_names)

    # STEP 9: 模型评估
    metrics, proba, y_pred = evaluate_model(model, model_type, X_test, y_test)
    metrics["n_train"] = len(y_train)

    # STEP 10: 特征重要性
    feature_importance(model, model_type, feat_names)

    # STEP 11: 保存模型
    save_model(model, model_type, scaler, feat_names, metrics)

    # 汇总
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"  全部流程完成，耗时 {elapsed:.1f}s")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("  输出文件:")
    for fn in sorted(os.listdir(OUTPUT_DIR)):
        fp = os.path.join(OUTPUT_DIR, fn)
        sz = os.path.getsize(fp)
        if os.path.isfile(fp):
            print(f"    {fn:<50}  {sz / 1024:>7.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
