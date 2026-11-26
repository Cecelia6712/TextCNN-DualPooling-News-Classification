# 中文新闻短文本分类（TextCNN 架构演进与优化）

本项目基于 PyTorch 从零构建 TextCNN，用于中文短文本 4 分类任务（科技、体育、财经、娱乐）。项目经历了从基础架构到**双池化（Dual Pooling）**的深度迭代，最终将测试集最高准确率从 84% 突破至 **96.88%**。

## 🌟 核心亮点 (Highlights)
1. **数据清洗与特征工程**：针对网络新闻噪声，设计了完整的清洗管线（剔除数字、去标点、抗 OOV 字级 Tokenize），将词频信息熵集中在实体词上。
2. **窗口级 Mask 机制**：突破常规 `padding_idx=0` 的限制，在卷积后使用 `-inf` Mask 阻断补齐零向量的空间污染。
3. **架构演进与 Dual Pooling**：对比了 Max Pooling、Attention Pooling，最终创新性地提出**双池化融合（Max + Attention）**。既保留了局部硬特征，又结合了全局软注意力，成功捕捉“华为”、“夺冠”等核心实体词。
4. **四层瓶颈诊断闭环**：引入 TF-IDF+LR 朴素 Baseline 对照，从数据、特征、模型、优化四个层级精准定位小样本下的“维数灾难”问题。

## 📊 实验结果 (Results)
采用 5-seed 独立训练，基础版与双池化版对比：
| 模型架构 | 池化方式 | 最高单次准确率 | 5-seed 均值 ± 标准差 |
| :--- | :--- | :--- | :--- |
| Baseline | Max Pooling | 90.62% | 0.8438 ± 0.0523 |
| Attention | Attention | 93.75% | 0.8500 ± 0.0606 |
| **Final** | **Dual Pooling** | **🔥 96.88%** | **0.8438 ± 0.0765** |
*(注：小样本下复杂架构存在方差膨胀，体现了经典的 Bias-Variance Tradeoff。)*

## 📁 目录结构
```text
├── data/               # 语料数据与清洗脚本
├── src/                # 核心代码
│   ├── vocab.py        # 字级 Tokenize 与词表构建
│   ├── dataset.py      # Dataset 与 8:2 划分
│   ├── model.py        # TextCNN (含 Dual Pooling 架构)
│   └── utils.py        # 绘图与指标计算
├── train.py            # 训练与测试主脚本 (支持 --multi-seed)
├── predict.py          # 交互预测脚本
├── diagnose.py         # 数据质量诊断
├── analyze.py          # 四层瓶颈诊断
└── baseline_tfidf.py   # TF-IDF 朴素对照实验