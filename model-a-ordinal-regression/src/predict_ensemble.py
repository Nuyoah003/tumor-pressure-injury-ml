"""
肿瘤患者压力性损伤风险预测 —— 综合最优模型推理
================================================
加载 04_best_model_ensemble.py 训练保存的集成模型，对新数据进行预测。

支持输入方式:
  1. Excel文件 (.xlsx)
  2. CSV文件 (.csv)
  3. Python DataFrame（代码内直接调用 predict 函数）

用法:
  python predict_ensemble.py 数据.xlsx
  python predict_ensemble.py 数据.csv
  或在代码中: from predict_ensemble import predict; result = predict(df)
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════
# 配置
# ════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "saved_models", "best_ensemble_model.pkl")

TARGET      = "风险等级"
RISK_LABELS = ["无风险(0)", "低风险(1)", "中风险(2)", "高风险(3)"]
K           = 4

NOMINAL_VARS    = ["肿瘤类型"]
CONTINUOUS_VARS = ["年龄", "住院时长", "白蛋白", "BMI"]

# 模型训练时使用的全部特征列名（训练后写入pkl，这里仅作校验参考）
# 实际特征列表以模型 pkl 中保存的 feat_names 为准


# ════════════════════════════════════════════════════════════
# 模型加载
# ════════════════════════════════════════════════════════════
def load_model(model_path=None):
    """加载集成模型，返回模型包字典"""
    path = model_path or MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件不存在: {path}")
    pkg = joblib.load(path)
    print(f"[模型加载] {path}")
    print(f"  基模型数量: {len(pkg['models'])}")
    print(f"  阈值偏移: {pkg['threshold_offset']:.4f}")
    print(f"  特征数量: {len(pkg['feat_names'])}")
    return pkg


# ════════════════════════════════════════════════════════════
# 数据预处理
# ════════════════════════════════════════════════════════════
def preprocess(df_raw, pkg):
    """
    对原始数据进行与训练时一致的预处理。
    输入: 原始DataFrame（需包含与训练时相同的特征列）
    输出: 预处理后的DataFrame（仅保留模型需要的特征列，顺序一致）
    """
    df = df_raw.copy()

    # 去掉无关列
    df = df.drop(columns=["ID", TARGET, "Braden总分"], errors="ignore")

    # 名义变量 one-hot 编码
    for col in NOMINAL_VARS:
        if col not in df.columns:
            continue
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=int)
        dummies = dummies.iloc[:, :-1]
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    df = df.astype(float)

    # 对齐特征列（补缺失、排顺序）
    feat_names = pkg["feat_names"]
    for col in feat_names:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feat_names]

    # 标准化连续变量
    cols = [c for c in CONTINUOUS_VARS if c in df.columns]
    df[cols] = pkg["scaler"].transform(df[cols])

    return df


# ════════════════════════════════════════════════════════════
# 预测
# ════════════════════════════════════════════════════════════
def predict(df_raw, model_path=None, return_proba=False):
    """
    对输入数据进行预测。

    参数:
        df_raw:    原始DataFrame（需包含与训练时相同的特征列）
        model_path: 模型文件路径（默认自动定位）
        return_proba: 是否返回概率矩阵

    返回:
        如果 return_proba=False: pandas Series，索引为输入DataFrame的索引，值为预测等级 (0/1/2/3)
        如果 return_proba=True: tuple (Series, DataFrame)，DataFrame包含4列概率
    """
    pkg = load_model(model_path)
    X = preprocess(df_raw, pkg)

    # 软投票：多模型概率取平均
    probas = []
    for clf in pkg["models"]:
        p = clf.predict_proba(X.values)
        probas.append(p)
    avg_proba = np.mean(probas, axis=0)

    # 应用阈值偏移
    offset = pkg["threshold_offset"]
    adjusted = avg_proba.copy()
    for i in range(K):
        adjusted[:, i] += offset * (K - 1 - 2 * i) / (K - 1)
    y_pred = adjusted.argmax(axis=1)

    result = pd.Series(y_pred, index=df_raw.index, name="预测风险等级")
    result_label = result.map({
        0: "无风险", 1: "低风险", 2: "中风险", 3: "高风险"
    })
    result_label.name = "风险等级"

    if return_proba:
        proba_df = pd.DataFrame(
            avg_proba, columns=[f"P(等级{i})" for i in range(K)],
            index=df_raw.index
        )
        return result_label, proba_df

    return result_label


# ════════════════════════════════════════════════════════════
# 命令行入口
# ════════════════════════════════════════════════════════════
def main():
    if len(sys.argv) < 2:
        print("用法: python predict_ensemble.py <数据文件.xlsx或.csv>")
        print("示例: python predict_ensemble.py 新患者数据.xlsx")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"文件不存在: {input_path}")
        sys.exit(1)

    # 读取数据
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path)
    print(f"\n[数据读取] {len(df):,} 条记录")

    # 预测
    y_pred, proba = predict(df, return_proba=True)

    # 输出结果
    result_df = pd.concat([y_pred, proba], axis=1)
    print(f"\n{'='*50}")
    print(f"  预测结果 (共 {len(result_df)} 条)")
    print(f"{'='*50}")

    # 统计
    counts = y_pred.value_counts().sort_index()
    for label in ["无风险", "低风险", "中风险", "高风险"]:
        n = counts.get(label, 0)
        pct = n / len(y_pred) * 100 if len(y_pred) > 0 else 0
        print(f"  {label}: {n} ({pct:.1f}%)")

    print(f"\n  详细结果:")
    print(result_df.to_string(max_rows=50))

    # 保存结果
    suffix = os.path.splitext(input_path)[1]
    out_path = input_path.replace(suffix, "_预测结果.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  结果已保存 -> {out_path}")


if __name__ == "__main__":
    main()
