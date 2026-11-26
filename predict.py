# predict.py
import sys
import torch
from data.corpus import CLASS_NAMES, clean_text
from src.vocab import tokenize, encode
from src.model import TextCNN
from src.dataset import MAX_LEN


def load(ckpt='outputs/seed_1/best.pt'):  # 【修改】指向最优的 Seed 1
    blob = torch.load(ckpt, map_location='cpu')
    # 【修改】实例化模型时，同步双池化架构的参数（kernel_sizes 等）
    model = TextCNN(len(blob['vocab']), len(CLASS_NAMES),
                    num_filters=16, kernel_sizes=(2, 3, 4), dropout=0.5)
    model.load_state_dict(blob['state'])
    model.eval()
    return model, blob['vocab']


def predict(model, vocab, text):
    cleaned = clean_text(text)
    ids, l = encode(tokenize(cleaned), vocab, MAX_LEN)
    x = torch.tensor([ids])
    l_t = torch.tensor([l])
    with torch.no_grad():
        probs = torch.softmax(model(x, l_t), 1)[0].tolist()
    pred = max(range(len(probs)), key=probs.__getitem__)
    return cleaned, CLASS_NAMES[pred], probs


if __name__ == '__main__':
    model, vocab = load()
    samples = [' '.join(sys.argv[1:])] if len(sys.argv) > 1 else [
        "华为P70即将发布 搭载鸿蒙NEXT系统",
        "梅西梅开二度 阿根廷3:0击败巴西#梅西#",
        "沪指收复3000点 北向资金净流入超50亿",
        "肖战新剧《藏海传》杀青 定档暑期🎬",
    ]
    for s in samples:
        cleaned, pred, probs = predict(model, vocab, s)
        top = sorted(zip(CLASS_NAMES, probs), key=lambda x: -x[1])[:3]
        print(f"\n原句: {s}\n清洗: {cleaned}\n预测: {pred}  |  "
              + ', '.join(f'{k}={v:.3f}' for k, v in top))