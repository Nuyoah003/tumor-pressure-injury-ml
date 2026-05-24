"""
=======================================================
Step 1：数据集划分脚本
开发人：李金城（算法开发组）
功能：读取清洗后的 .xlsx 文件，按分层抽样划分为 train.csv / test.csv
=======================================================
使用方法：
    python step1_split_data.py
    或指定文件路径：
    python step1_split_data.py --input cleaned_data.xlsx --ratio 0.7
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── 配置（按实际情况修改）──────────────────────────────────────────────────
INPUT_FILE  = "/Work/ljc/tumor/data/20260511更正后数据.xlsx"   # 清洗后的 xlsx 文件路径（数据处理组提供）
OUTPUT_DIR  = "/Work/ljc/tumor/data"                   # 输出目录
TARGET_COL  = "风险等级"             # 因变量列名
TRAIN_RATIO = 0.8                  # 训练集比例（0.7 → 7:3；0.8 → 8:2）
SEED        = 42

# ── 所有自变量列名（按数据定义原则文档）────────────────────────────────────
# 二分类变量（0/1 编码，胡云帆已处理）
BINARY_COLS = [
    "性别",         # 0=女, 1=男
    "民族",         # 0=汉族, 1=非汉族
    "是否入住ICU",  # 0=否, 1=是
    "放疗史",
    "化疗史",
    "营养风险",
    "血栓风险",
    "高血压",
    "糖尿病",
    "骨转移",
    "疼痛",
    "肿瘤转移",
    "激素治疗",
    "免疫抑制剂",
    "恶性积液",
    "电解质紊乱",
    "感染",
    "导管数目",     # 0=无, 1=有
]

# 肿瘤类型（无序多分类，胡云帆应完成 One-hot 编码）
# One-hot 列名格式：肿瘤类型_1, 肿瘤类型_2, ... 肿瘤类型_15（共15类）
TUMOR_OHE_COLS = [f"肿瘤类型_{i}" for i in range(1, 16)]

# 连续变量（Z-score 标准化，胡云帆已处理）
CONTINUOUS_COLS = [
    "年龄",
    "住院时长",
    "白蛋白",
    "BMI",
]


def parse_args():
    parser = argparse.ArgumentParser(description="数据集划分工具")
    parser.add_argument("--input",  default=INPUT_FILE,  help="输入 xlsx 文件路径")
    parser.add_argument("--output", default=OUTPUT_DIR,   help="输出目录")
    parser.add_argument("--ratio",  type=float, default=TRAIN_RATIO, help="训练集比例（默认0.7）")
    parser.add_argument("--seed",   type=int,   default=SEED,        help="随机种子（默认42）")
    return parser.parse_args()


def load_xlsx(path: str) -> pd.DataFrame:
    """读取 xlsx 文件，自动识别 Sheet。"""
    print(f"[读取] 正在读取文件：{path}")
    xl = pd.ExcelFile(path)
    print(f"[读取] 检测到 Sheet：{xl.sheet_names}")

    # 优先使用第一个 Sheet
    df = pd.read_excel(path, sheet_name=xl.sheet_names[0])
    print(f"[读取] 原始数据维度：{df.shape}")
    print(f"[读取] 列名列表：\n  {list(df.columns)}\n")
    return df


def validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    检查列完整性：
    · 因变量必须存在
    · 自动适配实际列名（处理空格、全角字符等差异）
    · 若肿瘤类型尚未 One-hot 编码，自动执行编码
    """
    # ── 1. 检查因变量 ──
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"未找到因变量列 '{TARGET_COL}'，请确认列名。\n当前列名：{list(df.columns)}"
        )

    # ── 2. 处理肿瘤类型：若存在原始列且尚无 One-hot 列，自动编码 ──
    raw_tumor_col = "肿瘤类型"
    if raw_tumor_col in df.columns and "肿瘤类型_1" not in df.columns:
        print(f"[预处理] 检测到原始肿瘤类型列，自动执行 One-hot 编码...")
        tumor_dummies = pd.get_dummies(
            df[raw_tumor_col].astype(int),
            prefix="肿瘤类型",
        )
        # 补齐缺失的类别列（确保 1~15 全部存在）
        for col in TUMOR_OHE_COLS:
            if col not in tumor_dummies.columns:
                tumor_dummies[col] = 0
        tumor_dummies = tumor_dummies[TUMOR_OHE_COLS]
        df = pd.concat([df.drop(columns=[raw_tumor_col]), tumor_dummies], axis=1)
        print(f"[预处理] One-hot 编码完成，新增列：{TUMOR_OHE_COLS[:3]}...")

    # ── 3. 打印各列缺失情况 ──
    all_feature_cols = BINARY_COLS + TUMOR_OHE_COLS + CONTINUOUS_COLS
    missing_cols = [c for c in all_feature_cols if c not in df.columns]
    if missing_cols:
        print(f"[警告] 以下列在数据中不存在，将被跳过（共 {len(missing_cols)} 列）：")
        for c in missing_cols:
            print(f"         · {c}")

    return df


def check_target_distribution(df: pd.DataFrame):
    """打印因变量分布，便于核查各类别样本量是否充足。"""
    dist = df[TARGET_COL].value_counts().sort_index()
    label_map = {0: "无风险", 1: "低风险", 2: "中风险", 3: "高风险"}
    print("[因变量] 各风险等级分布：")
    for k, v in dist.items():
        pct = v / len(df) * 100
        print(f"  {k}（{label_map.get(k, '?')}）: {v:>6} 条  ({pct:.1f}%)")
    print()


def select_feature_columns(df: pd.DataFrame) -> list:
    """返回实际存在的特征列名（按定义顺序）。"""
    all_feature_cols = BINARY_COLS + TUMOR_OHE_COLS + CONTINUOUS_COLS
    exist_cols = [c for c in all_feature_cols if c in df.columns]
    print(f"[特征] 实际使用特征数：{len(exist_cols)} 列")
    return exist_cols


def split_and_save(
    df: pd.DataFrame,
    feature_cols: list,
    train_ratio: float,
    output_dir: str,
    seed: int,
):
    """
    按分层抽样划分训练集与测试集，确保各风险等级比例一致。
    输出 train.csv 和 test.csv。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保留因变量 + 特征列
    keep_cols = feature_cols + [TARGET_COL]
    df_clean  = df[keep_cols].copy()

    # 去掉因变量为 NaN 的行
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=[TARGET_COL])
    after = len(df_clean)
    if before != after:
        print(f"[清理] 删除因变量缺失行：{before - after} 条")

    # 因变量转整数
    df_clean[TARGET_COL] = df_clean[TARGET_COL].astype(int)

    # ── 分层抽样划分 ──────────────────────────────────────────────────────
    test_ratio = 1.0 - train_ratio
    train_df, test_df = train_test_split(
        df_clean,
        test_size=test_ratio,
        stratify=df_clean[TARGET_COL],
        random_state=seed,
    )

    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    # ── 打印划分结果 ──────────────────────────────────────────────────────
    label_map = {0: "无风险", 1: "低风险", 2: "中风险", 3: "高风险"}
    print(f"[划分] 总样本：{len(df_clean)} 条")
    print(f"[划分] 训练集：{len(train_df)} 条（{train_ratio:.0%}）")
    print(f"[划分] 测试集：{len(test_df)}  条（{test_ratio:.0%}）")
    print()
    print("[验证] 各风险等级在训练集/测试集中的分布比例：")
    print(f"  {'等级':<10} {'训练集':<10} {'测试集':<10} {'原始':<10}")
    for k in sorted(df_clean[TARGET_COL].unique()):
        orig  = (df_clean[TARGET_COL] == k).mean()
        tr    = (train_df[TARGET_COL] == k).mean()
        te    = (test_df[TARGET_COL]  == k).mean()
        print(f"  {k}({label_map.get(k,'?'):<4}) {tr:.3f}      {te:.3f}      {orig:.3f}")
    print()

    # ── 保存 ──────────────────────────────────────────────────────────────
    train_path = os.path.join(output_dir, "train.csv")
    test_path  = os.path.join(output_dir, "test.csv")
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path,   index=False, encoding="utf-8-sig")
    print(f"[输出] train.csv 已保存：{train_path}  （{len(train_df)} 条）")
    print(f"[输出] test.csv  已保存：{test_path}   （{len(test_df)} 条）")

    # ── 保存特征列名列表（供 MLP 脚本读取，确保顺序一致）────────────────
    import json
    feat_path = os.path.join(output_dir, "feature_names.json")
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    print(f"[输出] feature_names.json 已保存：{feat_path}")


def main():
    args = parse_args()

    # 1. 读取
    df = load_xlsx(args.input)

    # 2. 列验证 & 自动处理
    df = validate_columns(df)

    # 3. 查看因变量分布
    check_target_distribution(df)

    # 4. 确定特征列
    feature_cols = select_feature_columns(df)

    # 5. 划分 & 保存
    split_and_save(df, feature_cols, args.ratio, args.output, args.seed)

    print("\n划分完成！请将 train.csv / test.csv / feature_names.json 交给 MLP 训练脚本使用。")


if __name__ == "__main__":
    main()