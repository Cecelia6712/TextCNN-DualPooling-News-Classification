\# TextCNN 中文新闻四分类（课程设计）



\## 运行

pip install torch jieba matplotlib scikit-learn numpy

python train.py          # 输出到 outputs/

python predict.py        # 交互预测（带默认示例）

python predict.py "自定义句子"



\## 目录

\- data/corpus.py   语料 + 清洗函数

\- src/vocab.py     分词/词表/编码

\- src/dataset.py   8:2 划分 + DataLoader

\- src/model.py     TextCNN（多尺度卷积 + 窗口级 padding mask）

\- src/utils.py     种子/画图/保存

\- train.py         训练 + 测试 + 保存

\- predict.py       交互预测



\## 课程要求映射

| 要求 | 位置 |

|---|---|

| 词表构建/定长编码 | src/vocab.py |

| TextCNN 建模 | src/model.py |

| 训练/测试 8:2 | train.py, src/dataset.py |

| 保存概率/图表 | outputs/pred\_probs.csv, \*.png, metrics.json |

| 交互预测 | predict.py |

| 清洗重难点 | data/corpus.py 中 clean\_text + 语料设计 |

