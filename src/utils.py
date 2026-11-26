# src/utils.py
import os, json, csv, random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei',
                                   'Noto Sans CJK SC', 'PingFang SC',
                                   'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def plot_curve(history, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history['loss'], marker='o', ms=3)
    ax[0].set_title('Train Loss')
    ax[1].plot(history['train_acc'], marker='o', ms=3, label='train')
    ax[1].plot(history['val_acc'], marker='s', ms=3, label='test')
    ax[1].set_title('Accuracy')
    ax[1].set_ylim(0, 1.02)
    ax[1].legend()
    for a in ax:
        a.grid(alpha=.3)
        a.set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_cm(y_true, y_pred, names, path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xlabel('Pred')
    ax.set_ylabel('True')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_preds(y_true, probs, preds, names, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['idx', 'true'] + [f'p_{c}' for c in names] + ['pred'])
        for i, (t, p, d) in enumerate(zip(y_true, probs, preds)):
            w.writerow([i, names[t]] + [f'{x:.4f}' for x in p] + [names[d]])


def save_wrong(texts, y_true, preds, probs, names, path):
    with open(path, 'w', encoding='utf-8') as f:
        for t, a, b, p in zip(texts, y_true, preds, probs):
            if a != b:
                f.write(json.dumps({
                    'text': t,
                    'true': names[a],
                    'pred': names[b],
                    'prob': {names[j]: round(float(p[j]), 4)
                             for j in range(len(names))}
                }, ensure_ascii=False) + '\n')