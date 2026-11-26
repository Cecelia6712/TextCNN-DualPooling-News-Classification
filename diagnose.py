# diagnose.py
"""数据质量诊断：对比旧清洗 v1 与新清洗 v2。自包含实现，可独立运行。"""
import re, random, unicodedata, statistics
from collections import Counter
from data.corpus import CORPUS, clean_text as clean_v2

try:
    import jieba; jieba.setLogLevel(20); HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False


def tokenize(text):
    if not text:
        return []
    toks = jieba.lcut(text) if HAS_JIEBA else list(text)
    return [w.strip() for w in toks if w.strip()]


# ===== 旧清洗 v1（保留下划线前缀，避免覆盖 v2） =====
_EMOJI_OLD = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF"
                        "\U0001F1E0-\U0001F1FF\uFE0F\u200D]+")
_URL_OLD   = re.compile(r'(?:https?://|www\.)\S+', re.I)
_AT_OLD    = re.compile(r'@[A-Za-z0-9_\u4e00-\u9fa5\-]{1,30}')
_TOPIC_OLD = re.compile(r'#([^#\n]{1,30})#')
_MP_OLD    = re.compile(r'([!?！？。，,.])\1+')
_KEEP_OLD  = re.compile(r'[^\u4e00-\u9fa5A-Za-z0-9，。！？、：；（）《》【】%\.\-/: ]')


def clean_v1(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKC', text)
    text = _URL_OLD.sub(' URLTOKEN ', text)
    text = _AT_OLD.sub(' USERTOKEN ', text)
    text = _TOPIC_OLD.sub(r' \1 ', text)
    text = _EMOJI_OLD.sub(' ', text)
    text = text.replace('\u3000', ' ')
    text = _MP_OLD.sub(r'\1', text)
    text = _KEEP_OLD.sub(' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def report(title, cleaner):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)

    texts = [cleaner(t) for t, _ in CORPUS]

    print("\n【1】清洗后样本（随机 5 条）")
    random.seed(0)
    for i in random.sample(range(len(texts)), 5):
        print(f"  · {texts[i]}")
        print(f"    tokens = {tokenize(texts[i])}")

    all_toks = []
    for t in texts:
        all_toks.extend(tokenize(t))
    cnt = Counter(all_toks)
    v1 = sum(1 for _, c in cnt.items() if c >= 1) + 2
    v2 = sum(1 for _, c in cnt.items() if c >= 2) + 2
    print(f"\n【2】词表  min_freq=1: {v1}   min_freq=2: {v2}")
    print(f"     token 总数: {len(all_toks)}   unique: {len(set(all_toks))}")

    print("\n【3】Top-20 高频 token:")
    print("    " + "  ".join(f"{w}×{c}" for w, c in cnt.most_common(20)))

    singles = [w for w, c in cnt.items() if c == 1]
    print(f"\n【4】只出现 1 次的 token 共 {len(singles)} 个，随机 15 个：")
    if singles:
        print("    " + "  ".join(random.sample(singles, min(15, len(singles)))))

    lens = [len(tokenize(t)) for t in texts]
    print(f"\n【5】token 长度  min={min(lens)}  max={max(lens)}  "
          f"mean={statistics.mean(lens):.1f}  median={statistics.median(lens)}")
    print(f"    ≤1 token: {sum(1 for l in lens if l <= 1)} 条   "
          f">24 token（会被截断）: {sum(1 for l in lens if l > 24)} 条")


if __name__ == '__main__':
    report("旧清洗 v1（原规则）", clean_v1)
    report("新清洗 v2（数字归一 + 去标点 + 小写）", clean_v2)