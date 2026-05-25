# 模型推理接口文档

> 版本：v1.0 | 日期：2026-05-25 | 负责人：白伟琪

---

## 1. 概述

`src/predict.py` 提供统一的肿瘤患者压力性损伤风险预测接口，支持两种模型、三种模式：

| 模型类型 | `--model` 参数 | 说明 |
|----------|:--------------:|------|
| Model A 综合集成 | `ensemble` | Top3 软投票+阈值优化，Kappa=0.2856 |
| Model A 高风险预警 | `high_risk` | XGBoost+序数代价权重，高风险 Recall=0.4972 |
| Model B MLP 神经网络 | `mlp` | 256→128→64，Kappa=0.1806，高风险 Recall=0.6102 |

---

## 2. 快速开始

### 2.1 环境依赖

```
python >= 3.8
numpy, pandas, scikit-learn
torch >= 1.9          # 仅 Model B 需要
joblib                # 仅 Model A 需要
openpyxl              # 仅 Excel 输入输出需要
```

### 2.2 命令行

```bash
# 单条预测（打印到控制台）
python src/predict.py --model ensemble --input 患者.xlsx

# 批量预测（保存为 CSV）
python src/predict.py --model mlp --input 患者.csv --output 结果.csv

# 批量预测（保存为 Excel）
python src/predict.py --model high_risk --input 患者.xlsx --output 结果.xlsx

# 查看模型元信息
python src/predict.py --model mlp --meta

# 使用自定义模型路径
python src/predict.py --model ensemble --model-path /path/to/custom.pkl --input 患者.xlsx
```

### 2.3 Python API

```python
from src.predict import Predictor

# 初始化（自动查找模型文件）
p = Predictor("mlp")

# 单条预测
result = p.predict({"年龄": 68, "白蛋白": 30.5, "BMI": 20.1, "肿瘤类型": 1, ...})
print(result)
# → {
#     "predicted_class": 1,
#     "risk_label": "低风险",
#     "probability": {"无风险": 0.72, "低风险": 0.15, "中风险": 0.08, "高风险": 0.05}
#   }

# 批量预测
import pandas as pd
df = pd.read_excel("患者.xlsx")
results = p.predict(df)
# → DataFrame: predicted_class, risk_label, proba_0_无风险, ...

# 获取概率矩阵
proba = p.predict_proba(df)
# → np.ndarray, shape (N, 4)
```

---

## 3. Predictor 类

### 3.1 构造函数

```python
Predictor(model_type="mlp", model_path=None)
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `model_type` | `str` | 是 | `"ensemble"` / `"high_risk"` / `"mlp"` |
| `model_path` | `str` | 否 | 自定义模型文件路径，覆盖默认查找 |

模型文件默认查找顺序：

- **ensemble / high_risk** → `models/best_ensemble_model.pkl` → `../model-a-ordinal-regression/saved_models/`
- **mlp** → `models/model_MLP_v2.pt` + `models/model_config_v2.json` → `../model-b-mlp/model/`

### 3.2 predict(data)

预测风险等级。

| 参数 | 类型 | 说明 |
|------|------|------|
| `data` | `dict` | 单条患者数据 |
| `data` | `list[dict]` | 多条患者数据 |
| `data` | `pd.DataFrame` | 批量患者数据 |
| `data` | `str` | Excel（.xlsx）或 CSV（.csv）文件路径 |

**返回值：**

- 单条输入 → `dict`：

```python
{
    "predicted_class": int,      # 0=无风险, 1=低风险, 2=中风险, 3=高风险
    "risk_label": str,           # "无风险" / "低风险" / "中风险" / "高风险"
    "probability": {             # 四个类别的概率
        "无风险": float,
        "低风险": float,
        "中风险": float,
        "高风险": float,
    }
}
```

- 多条/文件输入 → `pd.DataFrame`：

| 列名 | 说明 |
|------|------|
| `predicted_class` | 预测类别 0-3 |
| `risk_label` | 风险标签 |
| `proba_0_无风险` | 无风险概率 |
| `proba_1_低风险` | 低风险概率 |
| `proba_2_中风险` | 中风险概率 |
| `proba_3_高风险` | 高风险概率 |

### 3.3 predict_proba(data)

返回原始概率矩阵，不做类别决策。

| 参数 | 同 `predict()` |
|------|---------------|

**返回值：** `np.ndarray`，shape `(N, 4)`，列顺序：无风险、低风险、中风险、高风险。

### 3.4 metadata（属性）

```python
>>> p.metadata
{
    "model_type": "mlp",
    "model_class": "mlp",
    "feature_count": 37,
    "feature_names": ["年龄", "住院时长", "白蛋白", "BMI", ...],
    "risk_labels": ["无风险", "低风险", "中风险", "高风险"],
    "device": "cpu",
}
```

---

## 4. 输入数据格式

### 4.1 必需字段

| 字段 | 类型 | 示例值 |
|------|------|:------:|
| 年龄 | 数值 | 68 |
| 住院时长 | 数值 | 15 |
| 白蛋白 | 数值 | 30.5 |
| BMI | 数值 | 22.1 |
| 肿瘤类型 | 分类（数值/字符串） | 1 或 "肺癌" |
| 性别 | 二值 | 0/1 |
| 民族 | 二值 | 0/1 |
| 是否入住ICU | 二值 | 0/1 |
| 放疗史 | 二值 | 0/1 |
| 化疗史 | 二值 | 0/1 |
| 营养风险 | 二值 | 0/1 |
| 血栓风险 | 二值 | 0/1 |
| 高血压 | 二值 | 0/1 |
| 糖尿病 | 二值 | 0/1 |
| 骨转移 | 二值 | 0/1 |
| 疼痛 | 二值 | 0/1 |
| 肿瘤转移 | 二值 | 0/1 |
| 激素治疗 | 二值 | 0/1 |
| 免疫抑制剂 | 二值 | 0/1 |
| 恶性积液 | 二值 | 0/1 |
| 电解质紊乱 | 二值 | 0/1 |
| 感染 | 二值 | 0/1 |
| 导管数目 | 二值 | 0/1 |

共 22 个原始变量（Missing fields are filled with 0）。

### 4.2 文件格式

- **Excel**（`.xlsx`）：自动读取第一个 sheet
- **CSV**（`.csv`）：UTF-8-BOM 编码，逗号分隔

---

## 5. CLI 参数

```
python src/predict.py [OPTIONS]
```

| 参数 | 必须 | 默认值 | 说明 |
|------|:---:|--------|------|
| `--model` | 是 | — | `ensemble` / `high_risk` / `mlp` |
| `--input` | 是* | — | 输入文件路径 |
| `--output` | 否 | `None` | 输出文件路径，不指定则打印到控制台 |
| `--model-path` | 否 | `None` | 自定义模型文件路径 |
| `--meta` | 否 | `False` | 只打印模型元信息（此时不需要 `--input`） |

\* `--meta` 模式下不需要 `--input`。

---

## 6. 完整示例

### 6.1 临床场景：入院初筛

```python
from src.predict import Predictor

# 使用 Model B 做高召回筛查
screener = Predictor("mlp")

patients = [
    {"年龄": 72, "住院时长": 8, "白蛋白": 28.0, "BMI": 19.5, "肿瘤类型": 3, ...},
    {"年龄": 55, "住院时长": 3, "白蛋白": 40.2, "BMI": 24.0, "肿瘤类型": 1, ...},
]

results = screener.predict(patients)
high_risk_flags = results["predicted_class"] == 3
print(f"高风险患者数: {high_risk_flags.sum()}/{len(results)}")
```

### 6.2 临床场景：精判复核

```python
# 对筛查出的有风险患者，用 Model A 精判
refiner = Predictor("ensemble")
confirmed = refiner.predict(screened_patients)
```

### 6.3 批量处理 + 保存

```python
import pandas as pd

p = Predictor("ensemble")
df = pd.read_excel("住院患者_20260525.xlsx")
result_df = p.predict(df)
result_df.to_excel("预测结果_20260525.xlsx", index=False)
```

---

## 7. 错误处理

| 错误 | 原因 | 解决 |
|------|------|------|
| `ValueError: 不支持的模型类型` | `model_type` 不在注册表中 | 使用 `ensemble` / `high_risk` / `mlp` |
| `FileNotFoundError: 找不到模型文件` | 默认路径无模型文件 | 将模型放入 `models/` 或用 `--model-path` 指定 |
| `TypeError: 不支持的数据类型` | 传入了非 dict/list/DataFrame/路径 的数据 | 转换为支持的格式 |
| `ImportError: No module named 'torch'` | 使用 Model B 但未安装 PyTorch | `pip install torch` |
| `ImportError: No module named 'joblib'` | 使用 Model A 但未安装 joblib | `pip install joblib` |

---

## 8. 与 evaluate.py 的关系

| 功能 | `evaluate.py` | `predict.py` |
|------|:---:|:---:|
| 模型评估（混淆矩阵、Kappa、AUC） | ✅ | — |
| 两模型对比报告 | ✅ | — |
| 超参数优化 | ✅ | — |
| 单条/批量推理 | — | ✅ |
| 概率输出 | — | ✅ |
| 命令行接口 | ✅ | ✅ |
| Python API | ✅ | ✅ |

`predict.py` 是轻量推理接口，只依赖模型文件，不依赖训练代码。`evaluate.py` 是完整评估框架，包含指标计算、对比报告、超参搜索。
