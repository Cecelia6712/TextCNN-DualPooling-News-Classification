# train.py
import os, json, copy, argparse
import torch
from torch import nn
from data.corpus import CORPUS, CLASS_NAMES, clean_text
from src.vocab import build_vocab
from src.dataset import make_loaders
from src.model import TextCNN
from src.utils import (set_seed, plot_curve, plot_cm,
                       save_preds, save_wrong)

EPOCHS = 25
BS     = 16
LR     = 0.002
DROPOUT = 0.5   # 【微调】从 0.3 提升至 0.5，压制双池化带来的过拟合风险


def evaluate(model, loader, device, crit=None):
    model.eval()
    logits_all, y_all, loss_sum = [], [], 0.0
    with torch.no_grad():
        for x, l, y in loader:
            x, l, y = x.to(device), l.to(device), y.to(device)
            logits = model(x, l)
            if crit:
                loss_sum += crit(logits, y).item() * y.size(0)
            logits_all.append(logits)
            y_all.append(y)
    logits = torch.cat(logits_all)
    y = torch.cat(y_all)
    probs = torch.softmax(logits, 1).cpu().numpy()
    preds = logits.argmax(1).cpu().numpy()
    acc = float((preds == y.cpu().numpy()).mean())
    loss = loss_sum / len(loader.dataset) if crit else None
    return acc, loss, probs, y.cpu().numpy(), preds


def run_once(seed, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 数据准备
    texts = [clean_text(t) for t, _ in CORPUS]
    labels = [y for _, y in CORPUS]
    vocab = build_vocab(texts, min_freq=1)
    tr_loader, te_loader, te_texts, te_labels, n_tr, n_te = \
        make_loaders(texts, labels, vocab, BS, seed=seed)
    
    print(f"[Seed {seed}] 词表={len(vocab)}  训练={n_tr}  测试={n_te}")
    json.dump(vocab, open(f'{out_dir}/vocab.json', 'w', encoding='utf-8'),
              ensure_ascii=False)

    # 2. 模型初始化 (使用双池化架构 + 0.5 dropout)
    model = TextCNN(len(vocab), len(CLASS_NAMES),
                    num_filters=16, kernel_sizes=(2, 3, 4),
                    dropout=DROPOUT).to(device)
    
    # 3. 损失函数与优化器
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    optim = torch.optim.Adam(model.parameters(),
                             lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

    # 4. 训练循环
    history = {'loss': [], 'train_acc': [], 'val_acc': []}
    best_acc, best_state = -1, None
    for ep in range(1, EPOCHS + 1):
        model.train()
        ls, c, n = 0.0, 0, 0
        for x, l, y in tr_loader:
            x, l, y = x.to(device), l.to(device), y.to(device)
            optim.zero_grad()
            out = model(x, l)
            loss = crit(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            ls += loss.item() * y.size(0)
            c += (out.argmax(1) == y).sum().item()
            n += y.size(0)
            
        sched.step()
        val_acc, *_ = evaluate(model, te_loader, device)
        history['loss'].append(ls / n)
        history['train_acc'].append(c / n)
        history['val_acc'].append(val_acc)
        
        print(f"[{ep:02d}] loss={ls/n:.4f} train={c/n:.4f} test={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    # 5. 恢复最佳权重并评估
    model.load_state_dict(best_state)
    torch.save({'state': best_state, 'vocab': vocab}, f'{out_dir}/best.pt')
    test_acc, test_loss, probs, y_true, preds = evaluate(model, te_loader, device, crit)

    # 6. 保存结果与图表
    json.dump({'seed': seed, 'test_acc': test_acc, 'test_loss': test_loss,
               'vocab_size': len(vocab), 'train': n_tr, 'test': n_te,
               'history': history},
              open(f'{out_dir}/metrics.json', 'w'), indent=2)
              
    plot_curve(history, f'{out_dir}/learning_curve.png')
    plot_cm(y_true, preds, CLASS_NAMES, f'{out_dir}/confusion_matrix.png')
    save_preds(y_true, probs, preds, CLASS_NAMES, f'{out_dir}/pred_probs.csv')
    save_wrong(te_texts, y_true, preds, probs, CLASS_NAMES, f'{out_dir}/wrong_cases.txt')

    from sklearn.metrics import classification_report
    rep = classification_report(y_true, preds, target_names=CLASS_NAMES,
                                digits=4, zero_division=0)
    open(f'{out_dir}/classification_report.txt', 'w', encoding='utf-8').write(rep)
    print(f"[Seed {seed}] 最终 test_acc={test_acc:.4f}\n")
    return test_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--multi-seed', action='store_true')
    args = ap.parse_args()

    if args.multi_seed:
        import numpy as np
        accs = []
        for s in [1, 2, 3, 4, 5]:
            accs.append(run_once(s, out_dir=f'outputs/seed_{s}'))
        print("\n" + "=" * 56)
        print(f"5-seed 测试准确率: {[round(a, 4) for a in accs]}")
        print(f"均值 ± 标准差: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print("=" * 56)
    else:
        run_once(args.seed, out_dir='outputs')


if __name__ == '__main__':
    main()