# baseline_tfidf.py
"""
朴素 ML baseline：TF-IDF + Logistic Regression
作用：给出"数据本身能分到多高"的参考上限。
  - 若 ≥70% → 数据可分，TextCNN 需要优化（数据增强/双通道）
  - 若 <60% → 数据本身有隐噪，报告写数据量限制
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from data.corpus import CORPUS, CLASS_NAMES, clean_text
from src.dataset import split_8_2

texts = [clean_text(t) for t, _ in CORPUS]
labels = [y for _, y in CORPUS]

# 5 个 seed 跑，看均值±std（和 TextCNN 对比才有意义）
accs = []
for seed in [1, 2, 3, 4, 5]:
    tr, te = split_8_2(labels, seed=seed)
    X = TfidfVectorizer(analyzer='char', ngram_range=(1, 2)).fit_transform(texts)
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(
        X[tr], [labels[i] for i in tr])
    pred = clf.predict(X[te])
    y_true = [labels[i] for i in te]
    acc = accuracy_score(y_true, pred)
    accs.append(acc)
    print(f"[Seed {seed}] TF-IDF+LR test_acc = {acc:.4f}")

print("\n" + "=" * 56)
print(f"TF-IDF+LR 5-seed 准确率: {[round(a, 4) for a in accs]}")
print(f"均值 ± 标准差: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print("=" * 56)

# 用 seed=3 打印详细报告
tr, te = split_8_2(labels, seed=3)
X = TfidfVectorizer(analyzer='char', ngram_range=(1, 2)).fit_transform(texts)
clf = LogisticRegression(max_iter=1000).fit(X[tr], [labels[i] for i in tr])
pred = clf.predict(X[te])
print("\n[Seed 3] 详细分类报告：")
print(classification_report([labels[i] for i in te], pred,
                            target_names=CLASS_NAMES, digits=4))