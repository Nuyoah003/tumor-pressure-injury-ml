# Model Evaluation 运行说明

## 环境依赖

### Python 版本

Python 3.8+

### 必需依赖

| 包名 | 用途 | 必需 |
|------|------|------|
| `torch` | Model B（MLP）加载与推理 | 是 |
| `scikit-learn` | 指标计算（AUC/Kappa/F1） | 是 |
| `pandas` | 数据处理 | 是 |
| `numpy` | 数值计算 | 是 |
| `matplotlib` | 图表绘制 | 否（缺失则跳过图表） |
| `seaborn` | 热力图（混淆矩阵） | 否（缺失则跳过图表） |
| `joblib` | Model A pkl 加载 | 否（缺失回退到 pickle） |

### 安装命令

```bash
# 最小安装（仅评估，无图表）
pip install torch scikit-learn pandas numpy

# 完整安装（含图表）
pip install torch scikit-learn pandas numpy matplotlib seaborn joblib
```

---

## 目录结构

```
model-evaluation/
├── src/
│   └── evaluate.py               # 统一评估脚本
│   ├── predict.py                # 统一预测脚本封装接口
│   ├── api.md                    # 接口文档
├── models/                       # 模型文件存储
│   ├── best_ensemble_model.pkl   # Model A：最优集成模型
│   ├── high_risk_model.pkl       # Model A：高风险预警模型
│   ├── model_MLP_v2.pt           # Model B：PyTorch 完整模型
│   ├── model_MLP_v2_state_dict.pth
│   ├── model_MLP_v2.h5           # Model B：Keras 导出
│   └── model_config_v2.json      # Model B：配置与特征名
├── data/
│   └── feature_names.json        # 特征名列表
│   ├── test.csv                  # 测试集数据
    ├── train.csv                 # 训练集数据
├── reports/                      # 评估输出目录（运行时生成）
├── evaluation.md                 # 本文件
└── README.md
```

---

## 数据准备

### 前置条件

`evaluate.py` 依赖 `test.csv`（统一测试集）。该文件由 Model B 的 `process.py` 从原始 Excel 数据生成。

### 生成测试数据

```bash
# 方法1：运行 Model B 的划分脚本
cd model-b-mlp
python src/process.py --input 20260511更正后数据.xlsx --ratio 0.8

# 该脚本会在 model-b-mlp/data/ 下生成：
#   train.csv（训练集，~80%）
#   test.csv（测试集，~20%）
#   feature_names.json
```

### 验证数据就绪

```bash
# 确认 test.csv 存在
ls -lh model-b-mlp/data/test.csv
```

---

## 运行模式

### 模式一：评估 Model B 单模型

仅对 MLP 模型在测试集上做评估，输出指标和图表。

```bash
cd model-evaluation

python src/evaluate.py \
    --model-b model-b-mlp/model/model_MLP_v2.pt \
    --model-b-config model-b-mlp/model/model_config_v2.json \
    --data model-b-mlp/data/
```

**输出文件**（位于 `reports/`）：

| 文件 | 说明 |
|------|------|
| `model_b_metrics.json` | MLP 全部指标（JSON） |
| `model_b_confusion_matrix.png` | 混淆矩阵图 |
| `model_b_roc.png` | ROC 曲线图 |

---

### 模式二：评估 Model A 单模型

对有序逻辑回归集成模型在测试集上做评估。

```bash
cd model-evaluation

# 使用集成模型（推荐，综合性能最优）
python src/evaluate.py \
    --model-a model-a-ordinal-regression/saved_models/best_ensemble_model.pkl \
    --data model-b-mlp/data/

# 或使用高风险预警模型
python src/evaluate.py \
    --model-a model-a-ordinal-regression/saved_models/high_risk_model.pkl \
    --data model-b-mlp/data/
```

**输出文件**（位于 `reports/`）：

| 文件 | 说明 |
|------|------|
| `model_a_metrics.json` | Model A 全部指标（JSON） |
| `model_a_confusion_matrix.png` | 混淆矩阵图 |
| `model_a_roc.png` | ROC 曲线图 |

---

### 模式三：两模型完整对比评估

同时评估两个模型并生成对比报告。这是最常用的模式。

```bash
cd model-evaluation

python src/evaluate.py \
    --model-a model-a-ordinal-regression/saved_models/best_ensemble_model.pkl \
    --model-b model-b-mlp/model/model_MLP_v2.pt \
    --model-b-config model-b-mlp/model/model_config_v2.json \
    --data model-b-mlp/data/
```

**输出文件**（位于 `reports/`）：

| 文件 | 说明 |
|------|------|
| `model_a_metrics.json` | Model A 全部指标 |
| `model_b_metrics.json` | Model B 全部指标 |
| `model_a_confusion_matrix.png` | Model A 混淆矩阵 |
| `model_b_confusion_matrix.png` | Model B 混淆矩阵 |
| `model_a_roc.png` | Model A ROC 曲线 |
| `model_b_roc.png` | Model B ROC 曲线 |
| `comparison_metrics.csv` | 全指标对比表（Excel 可打开） |
| `comparison_report.md` | 论文格式文字报告（含结论） |
| `confusion_matrix_comparison.png` | 混淆矩阵并排对照 |
| `metrics_comparison.png` | 综合指标柱状图 |
| `roc_comparison.png` | ROC 曲线叠图（四类分面） |

---

### 模式四：推理预测

对单个患者或一批患者进行风险等级预测。

```bash
cd model-evaluation

# 使用 Model A 集成模型预测
python src/evaluate.py --predict \
    --model-a model-a-ordinal-regression/saved_models/best_ensemble_model.pkl \
    --input 患者数据.xlsx

# 使用 Model B MLP 模型预测
python src/evaluate.py --predict \
    --model-b model-b-mlp/model/model_MLP_v2.pt \
    --model-b-config model-b-mlp/model/model_config_v2.json \
    --input 患者数据.csv
```

> 输入文件格式：Excel（.xlsx）或 CSV（.csv），需包含全部原始特征列（性别、年龄、肿瘤类型等）。

**输出示例**：

```
低风险(1)  {'无风险(0)': 0.7234, '低风险(1)': 0.1511, '中风险(2)': 0.0821, '高风险(3)': 0.0434}
高风险(3)  {'无风险(0)': 0.1123, '低风险(1)': 0.2156, '中风险(2)': 0.1987, '高风险(3)': 0.4734}
```

---

### 模式五：超参数优化

对指定模型类型执行网格搜索超参数优化。

```bash
cd model-evaluation

# MLP 网格搜索（需同时有 train.csv 和 test.csv）
python src/evaluate.py --hyperopt mlp --data ../model-b-mlp/data/

# XGBoost 网格搜索
python src/evaluate.py --hyperopt xgboost --data ../model-b-mlp/data/

# LightGBM 网格搜索
python src/evaluate.py --hyperopt lightgbm --data ../model-b-mlp/data/
```

**输出文件**（位于 `reports/`）：

| 文件 | 说明 |
|------|------|
| `hyperparameter_search.csv` | MLP 搜索记录（所有配置的得分） |
| `hyperparameter_search_xgb.csv` | XGBoost 搜索记录 |
| `hyperparameter_search_lgb.csv` | LightGBM 搜索记录 |

**MLP 搜索空间**：

| 超参数 | 候选值 |
|--------|--------|
| `hidden_units` | [128,64] / [256,128,64] / [256,128] / [128,64,32] / [512,256,128] |
| `dropout_rate` | 0.2 / 0.3 / 0.4 |
| `learning_rate` | 1e-3 / 5e-4 |
| `batch_size` | 256 / 512 |

---

## 指标说明

`compute_metrics()` 输出的完整指标清单：

| 指标 | 类型 | 含义 |
|------|------|------|
| `accuracy` | 综合 | 整体准确率 |
| `kappa` | 综合 | Cohen's Kappa（一致性系数） |
| `auc_macro` | 综合 | 多分类 AUC（macro 平均） |
| `auc_weighted` | 综合 | 多分类 AUC（weighted 平均） |
| `macro_f1` | 综合 | Macro F1-score |
| `weighted_f1` | 综合 | Weighted F1-score |
| `g_mean` | 综合 | 各类别召回率几何平均 |
| `per_class[i].precision` | 各类别 | 第 i 类精确率 |
| `per_class[i].recall` | 各类别 | 第 i 类召回率 |
| `per_class[i].f1` | 各类别 | 第 i 类 F1-score |
| `per_class[i].support` | 各类别 | 第 i 类真实样本数 |
| `per_class_auc[i]` | 各类别 | 第 i 类 OVR AUC |
| `confusion_matrix` | 各类别 | 4×4 混淆矩阵 |

---

## 代码调用接口

```python
import sys
sys.path.insert(0, "model-evaluation/src")
from evaluate import load_model_a, load_model_b, predict_model_a, predict_model_b, compute_metrics, predict_single

# 加载模型
pkg_a = load_model_a("models/best_ensemble_model.pkl")
model_b, config_b = load_model_b("models/model_MLP_v2.pt", "models/model_config_v2.json")

# 单条推理
result = predict_single(pkg_a, {
    "性别": 1, "年龄": 68, "白蛋白": 30.5, "BMI": 20.1,
    "肿瘤类型": 1, "化疗史": 0, "营养风险": 1, "血栓风险": 0,
    # ... 其余特征填默认值即可
}, model_type="a")
# => {"predicted_class": 1, "risk_label": "低风险(1)",
#     "probability": {"无风险(0)": 0.7234, "低风险(1)": 0.1511, ...}}

# 计算指标
metrics = compute_metrics(y_true, y_pred, y_proba, model_name="MyModel")
print(metrics["kappa"], metrics["auc_macro"], metrics["macro_f1"])
```

---

## 输出文件路径速查

```bash
reports/
├── model_a_metrics.json              # Model A 指标 JSON
├── model_b_metrics.json              # Model B 指标 JSON
├── model_a_confusion_matrix.png      # Model A 混淆矩阵
├── model_b_confusion_matrix.png      # Model B 混淆矩阵
├── model_a_roc.png                   # Model A ROC 曲线
├── model_b_roc.png                   # Model B ROC 曲线
├── comparison_metrics.csv            # 对比指标表（CSV）
├── comparison_report.md              # 对比报告（Markdown）
├── confusion_matrix_comparison.png   # 混淆矩阵并排对比
├── metrics_comparison.png            # 综合指标柱状图
├── roc_comparison.png                # ROC 曲线叠图
```

---

## 常见问题

**Q: 运行时报 "测试数据不存在"**

A: 需要先生成 `test.csv`。运行 `model-b-mlp/src/process.py` 从原始 Excel 划分数据。

**Q: Model A 评估报错**

A: 确认 pkl 文件存在。如果 `saved_models/` 下无文件，需先运行 Model A 的训练脚本（`04_best_model_ensemble.py` 或 `05_high_risk_model.py`）。

**Q: 图表中文字符显示为方块**

A: 安装中文字体。Windows 通常已有 SimHei/Microsoft YaHei；Linux 执行 `sudo apt install fonts-wqy-zenhei`。

**Q: torch.load 报错**

A: 检查 PyTorch 版本。若模型在不同版本下保存，可能需加 `weights_only=False` 参数（较新版 PyTorch）。

**Q: 两模型特征维度不一致**

A: 这是正常的。Model A 使用 36 维（肿瘤类型 One-hot 去了最后一类作参照），Model B 使用 37 维（保留全部 15 类 One-hot）。`evaluate.py` 会根据各模型的 `feat_names` 自动做特征对齐。
