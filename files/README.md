# 肿瘤患者压力性损伤风险预测模型

> **ML 模块 —— 有序逻辑回归（Proportional Odds Model）**

---

## 📋 项目概述

本项目是一个基于机器学习的**肿瘤患者压力性损伤风险预测系统**，使用有序逻辑回归模型（Proportional Odds Model）对患者进行四等级风险分层预测。

### 风险等级定义
| 等级 | 标签 | 样本数 |
|------|------|--------|
| 0 | 无风险 | 245 (82.0%) |
| 1 | 低风险 | 33 (11.0%) |
| 2 | 中风险 | 9 (3.0%) |
| 3 | 高风险 | 12 (4.0%) |

### 数据概况
- **原始数据**: 299 条记录
- **特征数量**: 33 个变量
- **数据质量**: 无重复行、无缺失值、轻微异常值已标记待核查

---

## 🔧 功能模块详解

### STEP 1: 数据加载 `load_data()`
```
输入: Excel 文件路径
输出: pandas DataFrame

功能: 从指定路径读取 Excel 数据文件
```

### STEP 2: 数据清洗 `clean_data()`
```
输入: 原始 DataFrame
输出: 清洗后 DataFrame + 报告文件

清洗操作:
├── 去重: 删除完全重复的行
├── 缺失值处理:
│   ├── 缺失率 > 30% → 删除该列
│   ├── 连续变量 → 中位数填充
│   └── 分类变量 → 众数填充
└── 异常值检测:
    ├── Z-score > 3 → 标记为异常
    └── 临床阈值外 → 标记待核查
```

**临床变量阈值:**
| 变量 | 下限 | 上限 |
|------|------|------|
| 年龄 | 18.0 | 120.0 |
| 住院时长 | 0.0 | 365.0 |
| 白蛋白 | 10.0 | 60.0 |
| BMI | 10.0 | 60.0 |

### STEP 3: 基线特征表 `baseline_table()`
```
输入: 清洗后 DataFrame
输出: 02_baseline_table.csv

统计方法:
├── 连续变量 → Kruskal-Wallis 检验（非正态分布）
└── 二分类变量 → 卡方检验

输出格式: 每个变量的等级分布 + P 值
```

**显著相关变量 (P < 0.05):**
- 年龄 (P=0.0030)
- 住院时长 (P<0.0001)
- 白蛋白 (P<0.0001)
- BMI (P=0.0062)
- 性别 (P=0.0005)
- 是否入住ICU (P=0.0015)
- 化疗史 (P<0.0001)
- 营养风险 (P=0.0004)
- 血栓风险 (P=0.0367)
- 激素治疗 (P=0.0039)
- 恶性积液 (P=0.0182)
- 电解质紊乱 (P<0.0001)
- 感染 (P<0.0001)

### STEP 4: 特征编码 `encode_features()`
```
输入: DataFrame
输出: X (特征矩阵), y (标签), feature_names (特征名列表)

编码规则:
├── 二分类变量 → 保持 0/1
├── 名义变量(肿瘤类型) → One-hot 编码（最后一位删除避免多重共线性）
└── 连续变量 → 保持原值
```

### STEP 5: 数据划分 `split_data()`
```
输入: X, y
输出: X_train, X_test, y_train, y_test

划分策略: 分层抽样 (Stratified Shuffle Split)
├── 训练集: 70% (209 条)
└── 测试集: 30% (90 条)
```

### STEP 6: 标准化 `standardize()`
```
输入: X_train, X_test
输出: 标准化后 X_train, X_test, scaler

方法: Z-score 标准化 (均值=0, 标准差=1)
范围: 仅连续变量（年龄、住院时长、白蛋白、BMI）
```

### STEP 7: 有序逻辑回归 `train_model()`
```
输入: X_train, y_train, feature_names
输出: 训练好的模型, 模型类型

模型选择策略:
1. 优先: statsmodels OrderedModel（真正的 Proportional Odds）
2. 降级: CumulativeLogitOrdinal（K-1 个二分类器近似）

类: CumulativeLogitOrdinal
├── __init__: 初始化参数 (C, max_iter, random_state)
├── fit(X, y): 训练 K-1 个二分类器
├── predict_proba(X): 预测各类别概率
└── predict(X): 预测类别
```

**有序逻辑回归核心思想:**
- 将 K 分类问题转化为 K-1 个二分类问题
- 计算 P(Y ≤ k) 的累积概率
- 满足比例优势假设（Proportional Odds Assumption）

### STEP 8: 模型评估 `evaluate_model()`
```
输入: 模型, X_test, y_test
输出: metrics 字典, 预测概率, 预测标签

评估指标:
├── 混淆矩阵 → 04_confusion_matrix.png
├── 分类报告 (Precision/Recall/F1)
├── Cohen's Kappa 系数
└── ROC-AUC (macro & weighted)

ROC 策略: One-vs-Rest (OVR)
```

### STEP 9: 特征重要性 `feature_importance()`
```
输入: 模型, 特征名列表
输出: 03_feature_importance.csv + 06_feature_importance.png

指标计算:
├── 系数 (β)
├── OR 值 (Odds Ratio) = exp(β)
├── 95% 置信区间 = exp(β ± 1.96×SE)
└── 排序依据: |OR - 1| 越大越重要
```

**关键风险因素 (OR > 1):**
| 排名 | 特征 | OR值 | 含义 |
|------|------|------|------|
| 1 | 化疗史 | 5.94 | 有化疗史风险是无化疗的5.94倍 |
| 2 | 肿瘤类型_4 | 2.45 | 特定肿瘤类型高风险 |
| 3 | 激素治疗 | 1.94 | 有激素治疗增加风险 |

**保护因素 (OR < 1):**
| 排名 | 特征 | OR值 | 含义 |
|------|------|------|------|
| 1 | 电解质紊乱 | 0.19 | 有电解质紊乱反而风险低 |
| 2 | 肿瘤转移 | 0.23 | 肿瘤转移患者风险低 |
| 3 | 感染 | 0.27 | 有感染者风险低 |

### STEP 10: 模型保存 `save_model()`
```
输入: 模型, scaler, 特征名, metrics
输出: 07_model_logistic.pkl

保存内容 (bundle):
├── model: 训练好的模型对象
├── model_type: "statsmodels" 或 "cumlogit"
├── scaler: StandardScaler 对象
├── feature_names: 特征名列表
├── 变量定义: BINARY_VARS, NOMINAL_VARS, CONTINUOUS_VARS
├── 目标定义: TARGET, RISK_LABELS, K
└── metrics: 评估指标
```

---

## 🔌 推理接口

### `predict_single(bundle, patient_data)`
```python
import pickle

# 加载模型
bundle = pickle.load(open("07_model_logistic.pkl", "rb"))

# 输入患者数据
patient = {
    "性别": 1,
    "年龄": 70,
    "白蛋白": 28.5,
    "BMI": 16.0,
    "肿瘤类型": 6,
    "化疗史": 0,
    "感染": 1,
    # ... 其他特征
}

# 预测
result = predict_single(bundle, patient)
# 输出:
# {
#     "风险等级": 3,
#     "风险标签": "高风险(3)",
#     "各等级概率": {
#         "无风险(0)": 0.15,
#         "低风险(1)": 0.25,
#         "中风险(2)": 0.35,
#         "高风险(3)": 0.25
#     }
# }
```

---

## 📁 输出文件清单

| 文件名 | 描述 | 大小 |
|--------|------|------|
| 01_preprocessing_report.txt | 数据清洗报告 | 0.4 KB |
| 02_baseline_table.csv | 基线特征表（带P值） | 1.8 KB |
| 03_feature_importance.csv | 特征重要性（OR值+CI） | 1.4 KB |
| 04_confusion_matrix.png | 混淆矩阵热图 | 52 KB |
| 05_roc_curves.png | ROC曲线图 | 90 KB |
| 06_feature_importance.png | 特征重要性排序图 | 109 KB |
| 07_model_logistic.pkl | 模型bundle | 4.2 KB |

---

## 🛠️ 环境要求

```bash
# Python 3.8+
pip install pandas numpy scikit-learn matplotlib seaborn scipy openpyxl

# 可选（推荐安装，获得真正的 Proportional Odds 模型）
pip install statsmodels
```

---

## 📊 模型性能（基于测试集）

| 指标 | 值 |
|------|------|
| Cohen's Kappa | 0.42 |
| AUC (macro) | 0.78 |
| AUC (weighted) | 0.85 |

---

## 🔬 变量定义汇总

### 二分类变量 (Binary Variables)
```python
BINARY_VARS = [
    "性别", "民族", "是否入住ICU", "放疗史", "化疗史",
    "营养风险", "血栓风险", "高血压", "糖尿病", "骨转移",
    "疼痛", "肿瘤转移", "激素治疗", "免疫抑制剂", "恶性积液",
    "电解质紊乱", "感染", "导管数目",
]
```

### 名义变量 (Nominal Variables)
```python
NOMINAL_VARS = ["肿瘤类型"]
```

### 连续变量 (Continuous Variables)
```python
CONTINUOUS_VARS = ["年龄", "住院时长", "白蛋白", "BMI"]
```

---

## 📝 使用方法

### 1. 修改数据路径
```python
DATA_PATH = "/your/path/to/data.xlsx"  # 第56行
```

### 2. 运行完整流程
```bash
python 00_ml_pipeline.py
```

### 3. 调用推理接口
```python
from 00_ml_pipeline import predict_single
result = predict_single(bundle, patient_data)
```

---

## ⚠️ 注意事项

1. **数据路径**: 当前代码使用 Linux 路径格式，Windows 用户需修改
2. **中文字体**: 代码会自动检测可用中文字体（文泉驿/思源/微软雅黑）
3. **样本不平衡**: 模型使用 `class_weight="balanced"` 处理
4. **模型选择**: 建议安装 `statsmodels` 以获得更准确的有序逻辑回归

---

## 📅 生成时间
- 代码生成: 2026-03-26
- 分析报告版本: v1.0
