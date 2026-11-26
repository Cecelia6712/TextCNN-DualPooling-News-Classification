# src/dataset.py
import random
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from src.vocab import tokenize, encode

MAX_LEN = 24


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.ids, self.lens = [], []
        for t in texts:
            ids, l = encode(tokenize(t), vocab, MAX_LEN)
            self.ids.append(ids)
            self.lens.append(l)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (torch.tensor(self.ids[i]),
                torch.tensor(self.lens[i]),
                torch.tensor(self.labels[i]))


def split_8_2(labels, seed=3):
    """每类 8:2 分层划分。"""
    rng = random.Random(seed)
    by_cls = defaultdict(list)
    for i, y in enumerate(labels):
        by_cls[y].append(i)
    tr, te = [], []
    for _, idx in by_cls.items():
        idx = idx[:]
        rng.shuffle(idx)
        n_te = max(1, round(len(idx) * 0.2))
        te += idx[:n_te]
        tr += idx[n_te:]
    return tr, te


def make_loaders(texts, labels, vocab, batch_size=16, seed=3):
    tr, te = split_8_2(labels, seed=seed)

    def loader(idx, shuffle):
        ds = TextDataset([texts[i] for i in idx], [labels[i] for i in idx], vocab)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return (loader(tr, True), loader(te, False),
            [texts[i] for i in te], [labels[i] for i in te], len(tr), len(te))