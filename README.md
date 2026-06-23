# 多疾病风险预测与分析系统

基于三大核心模型的疾病预测系统：多因素权重评估、自适应加权集成学习、贝叶斯网络多疾病关联分析。

---

## 项目概述

本系统包含三个核心算法模型，分别解决疾病预测与风险评估的不同问题：

### 核心模型

| 模型 | 文件 | 问题描述 |
|------|------|----------|
| **多因素影响权重评估** | `weight_evaluation.py` | 量化各因素对疾病风险的影响，识别关键风险因素 |
| **自适应加权集成学习 (AWELM)** | `awelm.py` | 构建高精度疾病预测模型，自适应不同疾病特点 |
| **贝叶斯网络多疾病关联 (BNMDAP)** | `bnmdap.py` | 量化疾病共病关联，预测多种疾病同时发生的概率 |

---

## 项目结构

```
DpDa/
├── core/                              # 核心模型目录
│   ├── weight_evaluation.py            # 问题1: 多因素权重评估
│   ├── awelm.py                      # 问题2: 自适应加权集成学习
│   └── bnmdap.py                     # 问题3: 贝叶斯网络多疾病关联
│
├── web/                               # Web应用目录
│   ├── app.py                        # Flask应用主程序
│   ├── model_utilities.py            # 模型工具库
│   ├── model_calibration.py          # 概率校准
│   └── multi_disease_model.py       # 多疾病预测
│
├── data/                              # 数据集目录
│   ├── stroke.csv                    # 中风数据集
│   ├── heart.csv                     # 心脏病数据集
│   └── cirrhosis.csv                 # 肝硬化数据集
│
├── templates/                         # HTML模板
│   ├── weight_evaluation.html        # 权重评估页面
│   ├── awelm.html                    # AWELM模型页面
│   └── bnmdap.html                  # BNMDAP页面
│
├── static/                           # 静态资源
│   ├── style.css                    # 样式表
│   ├── script.js                     # JavaScript
│   └── images/                      # 图片资源
│
├── output/                           # 输出目录
│   ├── figures/                     # 生成的图表
│   ├── models/                      # 训练模型
│   └── processed_data/              # 处理后数据
│
├── requirements.txt                   # 依赖包
└── README.md                         # 本文件
```

---

## 快速开始

### 环境要求
- Python 3.8+
- 详见 `requirements.txt`

### 安装依赖

```bash
pip install -r requirements.txt
```

### 预生成模型检查点（可选，首次运行后自动跳过）

```bash
python generate_checkpoints.py
```

这会预训练所有模型并将结果缓存到 `output/checkpoints/` 目录（9 个 JSON 文件），避免每次启动时重复训练。

### 运行Web应用

```bash
start.bat
```

浏览器自动打开 `http://localhost:5000`

---

## 核心模型详解

### 1. 多因素影响权重评估模型 (Problem 1)

**文件**: `weight_evaluation.py`

**目标**: 量化每个因素对疾病风险的贡献度，识别关键风险因素

**综合权重公式**:
```
W_j = |β_j_full| × (1 - p_j) × (1 + ΔAUC_j)
```

其中:
- `|β_j_full|`: 多元回归系数的绝对值
- `p_j`: p值（统计显著性）
- `ΔAUC_j`: 移除特征j后AUC下降值（对模型性能的独立贡献）

**使用方法**:
```python
from weight_evaluation import FeatureWeightEvaluator

evaluator = FeatureWeightEvaluator(dataset_name='heart')
summary = evaluator.run_full_pipeline(
    df, 
    target_col='HeartDisease',
    categorical_cols=['Sex', 'ChestPainType'],
    numeric_cols=['Age', 'RestingBP', 'Cholesterol']
)
```

### 2. 自适应加权集成学习模型 (AWELM)

**文件**: `awelm.py`

**目标**: 构建高精度疾病预测模型，通过优化集成权重融合多个基础模型

**基础模型**: Logistic Regression, Random Forest, Gradient Boosting, SVM

**损失函数** (含平衡因子和正则化):
```
L(w) = -mean[β·y·log(p) + (1-y)·log(1-p)] + λ·H(w)
```
其中 `p = Σw_i·p_i`, `β = negatives/positives`

**使用方法**:
```python
from awelm import AdaptiveWeightedEnsemble

ensemble = AdaptiveWeightedEnsemble(dataset_name='stroke')
results = ensemble.run_full_pipeline(X, y)
```

### 3. 贝叶斯网络多疾病关联概率模型 (BNMDAP)

**文件**: `bnmdap.py`

**目标**: 量化疾病共病关联，预测多种疾病同时发生的概率

**核心公式**:
- 条件概率: `P(A|B) = P(A,B) / P(B)`
- 贝叶斯公式: `P(B|A) = P(A|B)·P(B) / P(A)`
- 相对风险: `RR = P(Disease|RiskFactor) / P(Disease)`

**使用方法**:
```python
from bnmdap import BayesianDiseaseNetwork

network = BayesianDiseaseNetwork()
network.estimate_prior_from_data(stroke_df, heart_df, cirrhosis_df)
result = network.predict(hypertension=1, heart_disease=0, cirrhosis=0)
```

---

## Web应用功能

系统提供交互式Web界面，包含以下页面：

### 1. 权重评估页面 (`/weight-evaluation`)
- 查看各特征的权重排名
- 可视化累积权重曲线
- 热力图对比单变量AUC与综合权重

### 2. AWELM模型页面 (`/awelm`)
- 基础模型性能对比
- 集成模型权重分布
- ROC曲线比较

### 3. BNMDAP页面 (`/bnmdap`)
- 贝叶斯网络结构图
- 疾病关联热力图
- 共病概率预测

---

## API接口

### 权重评估API
```
GET /api/weight-evaluation/<dataset>
```
参数: `dataset` = heart | stroke | cirrhosis

### AWELM API
```
GET/POST /api/awelm/<dataset>
```
参数: `dataset` = heart | stroke | cirrhosis

### BNMDAP预测API
```
POST /api/bnmdap/predict
Content-Type: application/json

{
    "hypertension": 0,
    "heart_disease": 0,
    "cirrhosis": 0
}
```

---

## 模型训练结果

### 1. 权重评估 (Weight Evaluation)

| 数据集 | 特征数 | Full AUC | 描述 |
|--------|--------|----------|------|
| heart | 11 | 0.9178 | 心脏病风险因素权重分析 |
| stroke | 10 | 0.8443 | 中风风险因素权重分析 |
| cirrhosis | 17 | 0.7493 | 肝硬化分期(>=3)权重分析 |

### 2. AWELM 集成学习

| 数据集 | 最佳模型 | Best AUC | Ensemble AUC | 不平衡比例 |
|--------|----------|----------|-------------|-----------|
| heart | Gradient Boosting | 0.9348 | 0.9298 | 1.24 |
| stroke | Logistic Regression | 0.8377 | 0.6114 | 19.54 |
| cirrhosis | SVM | 0.6833 | 0.6549 | 2.52 |

注：stroke 数据集存在严重类别不平衡（~5% 正例），SVM 的 Platt Scaling 在极端不平衡下性能下降。

### 3. BNMDAP 贝叶斯网络

| 先验概率 | 估计值 | 来源 |
|----------|--------|------|
| P(Stroke) | 4.87% | 数据估计 |
| P(Heart Disease) | 55.34% | 数据估计 |
| P(Advanced Cirrhosis) | 71.53% | 数据估计 |
| P(Hypertension) | 9.75% | 数据估计 |

---

## 数据集说明

| 数据集 | 样本量 | 目标变量 | 特点 |
|--------|--------|----------|------|
| stroke.csv | ~5,110 | 是否中风(二分类) | 类别不平衡 |
| heart.csv | ~918 | 是否心脏病(二分类) | 特征组合复杂 |
| cirrhosis.csv | ~418 | 肝硬化分期(1-4) | 回归任务 |

---

## 依赖包

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
statsmodels>=0.13.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
flask>=2.0.0
joblib>=1.1.0
```

---

## 作者

- 石韫琨
- 曾子航
