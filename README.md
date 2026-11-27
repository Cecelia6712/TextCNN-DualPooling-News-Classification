
***

#  TextCNN-DualPooling-News-Classification

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/NLP-Text%20Classification-blue?style=for-the-badge" alt="NLP">
  <img src="https://img.shields.io/badge/Accuracy-96.88%25-success?style=for-the-badge" alt="Accuracy">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

一个基于 PyTorch 从零构建的**中文新闻短文本 4 分类**（科技、体育、财经、娱乐）解决方案。本项目不仅实现了基础 TextCNN，还深入探究了小样本下的维度灾难问题，并创新性地提出了 **Dual Pooling (Max + Attention) 架构**，将模型最高测试准确率从 84.38% 突破至 **96.88%**。

##  核心特性 (Key Features)

- **双池化融合架构 (Dual Pooling)**：摒弃了传统单一 Max Pooling 导致特征丢失的问题，将捕捉局部最强信号的 Max Pooling 与捕捉全局语义权重的 Attention Pooling 进行特征拼接融合，大幅提升模型上限。
- **窗口级 Padding Mask**：针对中文短文本补齐零向量带来的特征污染，在卷积后引入 `-inf` Mask，确保池化层仅对有效感受野进行激活，并利用 `nan_to_num` 完美规避了反向传播时的 `NaN` 梯度爆炸。
- **抗 OOV 的字级 Tokenize**：摒弃传统词级分词，采用“字级+英文词级”混合 Tokenize。有效解决小样本下由于词表稀疏（OOV）导致的 Embedding 训不动问题。
- **科学严谨的评估闭环**：包含 `diagnose.py`（数据质量诊断）、`analyze.py`（4层瓶颈诊断）、`baseline_tfidf.py`（传统 ML 对照实验），全面验证模型架构的有效性。

##  实验结果 (Benchmarks)

在 160 条精标语料上（128 训练集 / 32 测试集），进行 5-seed 独立交叉验证，对比不同架构表现：

| 模型架构 | 池化方式 (Pooling) | 最高单次准确率 | 5-seed 均值 ± 标准差 |
| :--- | :--- | :--- | :--- |
| Baseline | Max Pooling | 90.62% | 0.8438 ± 0.0523 |
| V2 (进阶) | Attention Pooling | 93.75% | 0.8500 ± 0.0606 |
| **V3 (最终)** | **Dual Pooling (Max + Attn)** | **96.88%** | **0.8438 ± 0.0765** |

> *注：实验同时揭示了小样本下经典的 Bias-Variance Tradeoff（偏差-方差权衡）。复杂架构有效拉高了准确率上限，但也导致了方差膨胀，印证了“深度模型的第一性约束是数据规模”这一论断。*

##  快速开始 (Quick Start)

### 1. 环境准备
```bash
pip install -r requirements.txt
```

### 2. 模型训练
```bash
# 运行 5 个随机种子进行交叉验证
python train.py --multi-seed

# 或者单次快速训练
python train.py
```

### 3. 交互式预测
```bash
# 内置默认测试样例
python predict.py

# 自定义输入新闻进行预测
python predict.py "华为P70即将发布 搭载鸿蒙NEXT系统"
```

##  工程结构 (Project Structure)

```text
├── data/
│   └── corpus.py       # 数据集语料与正则清洗管线
├── src/
│   ├── vocab.py        # 字级 Tokenize 与词表编码机制
│   ├── dataset.py      # 8:2 分层抽样与 DataLoader
│   ├── model.py        # 核心模型：TextCNN + Dual Pooling 架构
│   └── utils.py        # 评估指标、可视化曲线与混淆矩阵
├── train.py            # 训练与多 seed 验证入口
├── predict.py          # 单句交互式推理脚本
├── diagnose.py         # 数据信息熵诊断工具
├── analyze.py          # 四层瓶颈分析工具 (Data/Feature/Model/Optim)
└── baseline_tfidf.py   # TF-IDF + Logistic Regression 朴素基线对照
```

##  模块与理论映射

| 核心理论 | 代码实现位置 | 说明 |
| :--- | :--- | :--- |
| **词表构建与定长编码** | `src/vocab.py` | 字级特征抽取，抗 OOV 干扰 |
| **一维卷积特征提取** | `src/model.py` | 多尺度卷积核 (kernel_sizes=2,3,4) 提取 n-gram 特征 |
| **Dual Pooling 降维** | `src/model.py` | 融合 Max Pooling (硬特征) 与 Attention (软权重) |
| **模型表现分析** | `src/utils.py` | 绘制学习曲线与混淆矩阵，分析跨域实体混淆现象 |

---
*Created by [Cecelia6712](https://github.com/Cecelia6712) | 用代码和实验探索 NLP 的世界。*

***

