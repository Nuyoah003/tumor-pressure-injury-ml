# 模型B：MLP 神经网络

**负责人**：李金城

构建多层感知机（MLP）神经网络模型，用于肿瘤患者压力性损伤风险等级的有序四分类预测。

## 职责

- [x] MLP 网络结构设计与实现
- [x] 有序分类损失函数设计
- [x] 训练流程与超参数调优
- [x] 特征权重贡献度提取与可视化（SHAP）
- [x] 模型文件导出（.pt / .pth / .h5）
- [ ] 与模型A进行统一评估对比

## 目录结构

```
model-b-mlp/
├── src/
│   ├── process.py                        # Step 1: 数据划分脚本
│   └── train.py                          # Step 2: MLP 模型定义、训练、评估与导出
├── data/
│   ├── train.csv                         # 训练集（80,403 条）
│   ├── test.csv                          # 测试集（20,101 条）
│   └── feature_names.json                # 37 个特征名称列表
├── model/
│   ├── model_MLP_v2.pt                   # PyTorch 完整模型
│   ├── model_MLP_v2_state_dict.pth       # PyTorch state dict
│   ├── model_MLP_v2.h5                   # TensorFlow/Keras 导出
│   ├── model_config_v2.json              # 模型架构配置 + 特征名
│   ├── test_metrics_mlp.json             # 测试集评估指标
│   ├── test_confusion_matrix_mlp.csv     # 混淆矩阵
│   ├── feature_importance_mlp.csv        # SHAP 特征重要性
│   ├── feature_importance_mlp.png        # SHAP 重要性图
│   ├── feature_importance_mlp_weight.png # 权重法重要性图
│   ├── training_curves.png               # 训练曲线（Loss/Accuracy/F1）
│   └── test_evaluation_mlp.png           # 混淆矩阵 + ROC 曲线
├── output_mlp/                           # 输出目录
└── README.md
```

## 技术说明

### 模型架构

```
Input(37)
  -> Linear(256) -> BatchNorm1d -> ReLU -> Dropout(0.3)
  -> Linear(128) -> BatchNorm1d -> ReLU -> Dropout(0.3)
  -> Linear(64)  -> BatchNorm1d -> ReLU -> Dropout(0.3)
  -> Linear(4)   # 输出 4 类风险等级
```

### 输入特征（37 维）

| 类别 | 数量 | 示例 |
|------|------|------|
| 二值临床指标 | 18 | 性别、ICU 入住、放疗史、化疗史、营养风险、血栓风险、高血压、糖尿病等 |
| One-Hot 肿瘤类型 | 15 | 肿瘤类型 1-15 |
| 连续变量 | 4 | 年龄、住院天数、白蛋白、BMI |

### 目标变量

4 类有序风险等级：

| 等级 | 含义 | 训练集占比 |
|------|------|-----------|
| 0 | 无风险 | 95.8% |
| 1 | 低风险 | 3.1% |
| 2 | 中风险 | 0.5% |
| 3 | 高风险 | 0.6% |

### 训练策略

- **类别不平衡处理**：训练损失使用逆频率加权 `CrossEntropyLoss`，验证损失使用无权版本
- **优化器**：Adam (lr=1e-3, weight_decay=1e-4)
- **学习率调度**：`ReduceLROnPlateau`，监控验证集 macro-F1，factor=0.5, patience=8
- **早停**：监控验证集 macro-F1，patience=20
- **梯度裁剪**：max_norm=5.0
- **Batch Size**：512
- **随机种子**：42

### 测试集性能

| 指标 | 值 |
|------|-----|
| Accuracy | 81.4% |
| Macro F1 | 0.334 |
| Weighted F1 | 0.872 |
| Cohen's Kappa | 0.181 |
| AUC (macro) | 0.896 |
| AUC (weighted) | 0.902 |

### 特征重要性（SHAP Top-8）

| 排名 | 特征 | SHAP 值 |
|------|------|---------|
| 1 | 年龄 | 0.354 |
| 2 | 营养风险 | 0.329 |
| 3 | 血栓风险 | 0.302 |
| 4 | 白蛋白 | 0.261 |
| 5 | 化疗史 | 0.242 |
| 6 | 住院天数 | 0.182 |
| 7 | BMI | 0.112 |
| 8 | 激素治疗 | 0.097 |

## 使用方法

### 环境依赖

```
torch
tensorflow
scikit-learn
shap
pandas
numpy
matplotlib
openpyxl
```

### 运行步骤

```bash
# Step 1: 数据划分（原始 Excel -> train.csv / test.csv）
python src/process.py

# Step 2: 模型训练、评估与导出
python src/train.py
```

## 交付物

| 文件 | 格式 | 说明 |
|------|------|------|
| `model_MLP_v2.pt` | PyTorch | 完整模型，可直接 `torch.load()` 加载 |
| `model_MLP_v2_state_dict.pth` | PyTorch | state dict，需配合模型类定义使用 |
| `model_MLP_v2.h5` | Keras/HDF5 | TensorFlow 生态部署用 |
| `model_config_v2.json` | JSON | 模型超参数 + 特征名，供部署团队解析 |
