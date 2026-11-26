# analyze.py
"""
数据分析师式瓶颈诊断：一次性跑完 4 个诊断，定位瓶颈层级。
用法: python analyze.py
"""
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from collections import Counter

from data.corpus import CORPUS, CLASS_NAMES, clean_text
from src.vocab import build_vocab, tokenize, encode
from src.dataset import TextDataset, split_8_2, MAX_LEN
from src.model import TextCNN
from src.utils import set_seed

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics import mutual_info_score
    HAS_SK = True
except ImportError:
    HAS_SK = False


def sep(s='='):
    print(s * 72)


# ============================================================
# 【诊断 1】L1 数据层：任务本身可分吗？
#   用 TF-IDF 向量化句子 → 算 4 个类中心的余弦相似度
#   判断: > 0.85 语义重叠严重(任务难)，< 0.5 任务好分(问题在模型)
# ============================================================
def diag_1_class_separability():
    sep()
    print("【诊断 1】L1 数据层 — 类间可分性（TF-IDF + 类中心余弦）")
    sep()
    if not HAS_SK:
        print("  [!] 未装 scikit-learn，跳过")
        return

    texts = [clean_text(t) for t, _ in CORPUS]
    labels = np.array([y for _, y in CORPUS])
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
    X = tfidf.fit_transform(texts)

    centers = [X[labels == c].mean(axis=0) for c in range(4)]
    centers = np.vstack([np.asarray(c).flatten() for c in centers])
    sim = cosine_similarity(centers)

    print(f"\n  类间余弦相似度矩阵:")
    print(f"  {'':>6s}" + "".join(f"{n:>8s}" for n in CLASS_NAMES))
    for i, n in enumerate(CLASS_NAMES):
        print(f"  {n:>6s}" + "".join(f"{sim[i, j]:>8.3f}" for j in range(4)))

    off = sim[~np.eye(4, dtype=bool)]
    print(f"\n  类间平均相似度: {off.mean():.3f}   最大: {off.max():.3f}")
    if off.mean() > 0.85:
        print("  → 判断: 语义重叠严重，任务本身难分（L1 问题）")
    elif off.mean() > 0.5:
        print("  → 判断: 可分但有难度，模型有发挥空间（继续看 L2）")
    else:
        print("  → 判断: 类间非常可分，问题在模型/优化（往下看 L3/L4）")


# ============================================================
# 【诊断 2】L2 特征层：清洗后有没有丢掉判别信号？
#   对每个 token 计算它与 label 的互信息 → 找出每类判别力 Top-K
#   判断: 判别 token 是语义词(如"梅西/华为") → 特征好
#        判别 token 是标点/数字 → 清洗不够
# ============================================================
def diag_2_token_discrimination():
    sep()
    print("【诊断 2】L2 特征层 — 每类的 Top-8 判别 token（互信息）")
    sep()

    texts = [clean_text(t) for t, _ in CORPUS]
    labels = [y for _, y in CORPUS]

    tok2labels = {}
    for t, y in zip(texts, labels):
        for tok in set(tokenize(t)):
            tok2labels.setdefault(tok, []).append(y)

    tok_mi = {}
    for tok, ys in tok2labels.items():
        if len(ys) < 3:
            continue
        y_full = [0] * 4
        for y in ys:
            y_full[y] += 1
        # 与全集 labels 的互信息
        base = np.bincount(labels, minlength=4)
        mi = 0.0
        for c in range(4):
            if y_full[c] > 0 and base[c] > 0:
                p_tc = y_full[c] / len(ys)
                p_c = base[c] / len(labels)
                mi += p_tc * np.log(p_tc / p_c)
        tok_mi[tok] = mi

    top = sorted(tok_mi.items(), key=lambda x: -x[1])[:32]
    print("\n  全语料 Top-32 判别 token（按 MI 降序）：")
    row = []
    for i, (w, mi) in enumerate(top):
        row.append(f"{w}({mi:.2f})")
        if (i + 1) % 8 == 0:
            print("    " + "  ".join(row)); row = []
    if row:
        print("    " + "  ".join(row))

    # 每类的判别 token
    print("\n  每类专属判别 Top-8：")
    for c in range(4):
        scores = {}
        for tok, ys in tok2labels.items():
            if len(ys) < 2:
                continue
            score = ys.count(c) / len(ys) * np.log(len(ys) + 1)
            if score > 0:
                scores[tok] = score
        top_c = sorted(scores.items(), key=lambda x: -x[1])[:8]
        print(f"    {CLASS_NAMES[c]:>4s}: " + "  ".join(f"{w}" for w, _ in top_c))


# ============================================================
# 【诊断 3】L3 模型层：给 8 条样本，模型能训到 100% 吗？
#   这是"过拟合能力测试"(sanity check)：
#   能 → 模型/优化链路健康，问题在数据量
#   不能 → 优化链路有问题（lr 太小/太大、mask 有 bug、loss 配错）
# ============================================================
def diag_3_overfit_capacity():
    sep()
    print("【诊断 3】L3 模型层 — 小样本过拟合能力测试（8 条样本训到 100%）")
    sep()

    set_seed(0)
    device = 'cpu'
    texts = [clean_text(t) for t, _ in CORPUS]
    labels = [y for _, y in CORPUS]
    vocab = build_vocab(texts, min_freq=1)

    # 每类取 2 条，共 8 条
    rng = random.Random(0)
    idx_per_c = {c: [] for c in range(4)}
    for i, y in enumerate(labels):
        idx_per_c[y].append(i)
    small_idx = []
    for c in range(4):
        small_idx += rng.sample(idx_per_c[c], 2)

    ds = TextDataset([texts[i] for i in small_idx],
                     [labels[i] for i in small_idx], vocab)
    loader = DataLoader(ds, batch_size=8, shuffle=True)

    model = TextCNN(len(vocab), num_classes=4,
                    num_filters=16, dropout=0.0).to(device)  # 关掉 dropout
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()

    first_100 = None
    for ep in range(1, 201):
        model.train()
        for x, l, y in loader:
            optim.zero_grad()
            out = model(x, l)
            loss = crit(out, y)
            loss.backward()
            optim.step()
        # 检查 acc
        model.eval()
        with torch.no_grad():
            acc = 0
            for x, l, y in loader:
                out = model(x, l)
                acc += (out.argmax(1) == y).sum().item()
            acc /= len(ds)
        if acc == 1.0 and first_100 is None:
            first_100 = ep
        if ep in (20, 50, 100, 200):
            print(f"  epoch={ep:3d}  train_acc={acc:.4f}")

    print()
    if first_100 is None:
        print("  ❌ 200 epoch 仍训不到 100% —— 优化链路有问题（L4 问题）")
        print("     检查: lr 是否太小/太大、mask_windows 是否有 bug、loss 函数是否配错")
    elif first_100 <= 30:
        print(f"  ✅ {first_100} epoch 就训到 100% —— 优化链路健康")
        print("     → 模型容量足够，问题在数据量/特征（回到 L1/L2）")
    else:
        print(f"  ⚠️  需要 {first_100} epoch 才训到 100% —— 优化偏慢")
        print("     检查 lr 是否偏小、mask 是否过度过滤")


# ============================================================
# 【诊断 4】L4 + 全局：真实测试集错例的语义归因
#   训练一个完整模型，跑测试集，看错的样本"错在哪类之间"
#   判断: 错在相似类 → 承认边界模糊
#        错在不相干类 → 模型没学到 (回到 L2/L3)
# ============================================================
def diag_4_error_attribution():
    sep()
    print("【诊断 4】全局 — 测试集错例归因（错误的类间流向）")
    sep()

    set_seed(3)
    device = 'cpu'
    texts = [clean_text(t) for t, _ in CORPUS]
    labels = [y for _, y in CORPUS]
    vocab = build_vocab(texts, min_freq=1)
    tr_idx, te_idx = split_8_2(labels, seed=3)

    tr_loader = DataLoader(TextDataset([texts[i] for i in tr_idx],
                                       [labels[i] for i in tr_idx], vocab),
                            batch_size=16, shuffle=True)
    te_loader = DataLoader(TextDataset([texts[i] for i in te_idx],
                                       [labels[i] for i in te_idx], vocab),
                            batch_size=16, shuffle=False)

    model = TextCNN(len(vocab), num_classes=4,
                    num_filters=16, dropout=0.3)
    optim = torch.optim.Adam(model.parameters(), lr=0.002)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    for _ in range(25):
        model.train()
        for x, l, y in tr_loader:
            optim.zero_grad()
            loss = crit(model(x, l), y)
            loss.backward()
            optim.step()

    model.eval()
    confusion = np.zeros((4, 4), dtype=int)
    with torch.no_grad():
        for x, l, y in te_loader:
            pred = model(x, l).argmax(1)
            for yt, yp in zip(y.tolist(), pred.tolist()):
                confusion[yt, yp] += 1

    print("\n  测试集混淆矩阵（行=真，列=预）:")
    print(f"  {'':>6s}" + "".join(f"{n:>8s}" for n in CLASS_NAMES))
    for i, n in enumerate(CLASS_NAMES):
        print(f"  {n:>6s}" + "".join(f"{confusion[i, j]:>8d}" for j in range(4)))

    # 错例流向
    wrong_flow = Counter()
    for i in range(4):
        for j in range(4):
            if i != j and confusion[i, j] > 0:
                wrong_flow[(i, j)] = confusion[i, j]
    print("\n  错误流向 Top-5：")
    for (i, j), cnt in wrong_flow.most_common(5):
        print(f"    {CLASS_NAMES[i]:>4s} → {CLASS_NAMES[j]:>4s} : {cnt} 条")

    # 语义距离检查
    if HAS_SK:
        tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
        X = tfidf.fit_transform(texts)
        labels_arr = np.array(labels)
        centers = [np.asarray(X[labels_arr == c].mean(axis=0)).flatten()
                   for c in range(4)]
        centers = np.vstack(centers)
        print("\n  错误流向的类间 TF-IDF 距离（越低=语义越近=可接受）:")
        for (i, j), cnt in wrong_flow.most_common(3):
            d = np.linalg.norm(centers[i] - centers[j])
            verdict = "🟢 语义接近，难分" if d < 3 else "🟡 中等距离" if d < 5 else "🔴 语义远，说明模型没学到"
            print(f"    {CLASS_NAMES[i]} → {CLASS_NAMES[j]}: 距离={d:.2f}  {verdict}")


if __name__ == '__main__':
    print("\n" + "=" * 72)
    print("  数据分析师瓶颈诊断 · 4 层定位")
    print("=" * 72)
    diag_1_class_separability()
    diag_2_token_discrimination()
    diag_3_overfit_capacity()
    diag_4_error_attribution()
    print("\n" + "=" * 72)
    print("  诊断完成。对照下方『处方对照表』决定下一步。")
    print("=" * 72 + "\n")