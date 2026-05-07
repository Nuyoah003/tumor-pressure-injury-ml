# 模型A：有序逻辑回归（Ordinal Logistic Regression）

**负责人**：杨旭东

基于 Proportional Odds 假设构建有序逻辑回归模型（statsmodels OrderedModel），用于预测肿瘤患者的压力性损伤风险等级。

## 目录结构

```
model-a-ordinal-regression/
├── src/                          # Python 源码
│   ├── ml_pipeline.py            # 主流程：数据预处理 → 基线模型 → 评估
│   └── smote_optimization.py     # SMOTE 过采样优化实验
├── reports/                      # 报告与图表
│   ├── 13_数据清洗质量确认报告.md
│   ├── 14_模型开发技术报告.md
│   └── figures/                  # 可视化图表
│       ├── 04_confusion_matrix.png
│       ├── 05_roc_curves.png
│       ├── 06_feature_importance.png
│       ├── 10_roc_comparison.png
│       ├── 11_confusion_matrix_comparison.png
│       └── 12_metrics_comparison.png
└── README.md
```

## 当前进度

- [x] 数据预处理流程（清洗、编码、标准化、分层划分）
- [x] 有序逻辑回归基线模型
- [x] OR值 + 95% CI + P值计算
- [x] 特征重要性排序
- [x] 基线特征表（含 K-W 检验 P值）
- [x] SMOTE 过采样优化实验（对比 XGBoost / LightGBM）
- [ ] 进一步优化（待与甲方沟通 Braden 原始评分数据）

## 关键结果

| 指标 | 基线 OLR | OLR+SMOTE | XGBoost+SMOTE |
|------|----------|-----------|---------------|
| Cohen's Kappa | 0.078 | 0.210 | 0.176 |
| 高风险 Recall | 0% | 57.1% | 42.9% |
| G-mean | 0.199 | 0.336 | 0.402 |

## 运行方式

```bash
cd model-a-ordinal-regression/src
python ml_pipeline.py      # 运行主流程
python smote_optimization.py  # 运行SMOTE优化实验
```

> **注意**：运行前需将 `清洗后数据.xlsx` 放置到正确路径。数据文件因体积较大不纳入版本控制。
