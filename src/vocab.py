# src/vocab.py
# 字级 + 英文词级混合 tokenize（中文短文本 TextCNN 经典配置）
import re
from collections import Counter

PAD, UNK = 0, 1


def tokenize(text):
    """
    中文逐字 + 英文词级：
      - 华为 → 华 / 为
      - Mate → mate（整体保留，避免 m/a/t/e 分裂）
      - 数字/标点/空格丢弃
    """
    if not text:
        return []
    tokens = re.findall(r'[A-Za-z]+|[\u4e00-\u9fa5]', text)
    return [t.lower() for t in tokens if t]


def build_vocab(texts, min_freq=1):
    vocab = {'<PAD>': PAD, '<UNK>': UNK}
    cnt = Counter()
    for t in texts:
        cnt.update(tokenize(t))
    for w, c in cnt.most_common():
        if c >= min_freq:
            vocab[w] = len(vocab)
    return vocab


def encode(tokens, vocab, max_len):
    ids = [vocab.get(w, UNK) for w in tokens][:max_len] or [UNK]
    l = len(ids)
    return ids + [PAD] * (max_len - l), l