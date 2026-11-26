# src/model.py
import torch
from torch import nn

def mask_windows(h, l, k):
    """卷积输出 [B,C,L] 中窗口越界处填 -inf，池化自动忽略"""
    L = h.size(2)
    idx = torch.arange(L, device=h.device).unsqueeze(0)
    valid = (idx + k) <= l.unsqueeze(1)          # [B, L]
    valid[~valid.any(1)] = True                  # l<k 时退化为允许全部
    return h.masked_fill(~valid.unsqueeze(1), float('-inf'))

class TextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes=4,
                 embed_dim=32, num_filters=16,
                 kernel_sizes=(2, 3, 4), dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k)
                                    for k in kernel_sizes])
        self.attn_weights = nn.ModuleList([nn.Linear(num_filters, 1, bias=False) 
                                           for _ in kernel_sizes])
        self.drop = nn.Dropout(dropout)
        
        # 特征维度翻倍（Max + Attention）
        self.fc = nn.Linear(num_filters * len(kernel_sizes) * 2, num_classes)

    def forward(self, x, l):
        x = self.emb(x).transpose(1, 2) 
        feats = []
        for conv, attn in zip(self.convs, self.attn_weights):
            h = torch.relu(conv(x))              # [B, F, L_out]
            h = mask_windows(h, l, conv.kernel_size[0]) 
            
            # 【安全修复】把 -inf 转成极小值，杜绝 NaN 的产生！
            h_safe = torch.nan_to_num(h, neginf=-1e4)
            
            # 1. 常规的 Max Pooling (取最强特征)
            h_max = h_safe.max(dim=2).values  # [B, F]
            
            # 2. Attention Pooling (加权特征)
            h_t = h_safe.transpose(1, 2)         # [B, L_out, F]
            scores = attn(h_t).squeeze(-1)       # [B, L_out]

            # 生成 mask，防止对补齐位置分配注意力
            L_out = scores.size(1)
            idx = torch.arange(L_out, device=x.device).unsqueeze(0)
            valid = (idx + conv.kernel_size[0]) <= l.unsqueeze(1)
            valid[~valid.any(1)] = True
            
            scores = scores.masked_fill(~valid, -1e9)
            attn_weights = torch.softmax(scores, dim=-1).unsqueeze(-1) # [B, L_out, 1]
            h_attn = torch.sum(h_t * attn_weights, dim=1)              # [B, F]
            
            # 3. 拼接双池化特征
            feats.append(torch.cat([h_max, h_attn], dim=1))            # [B, 2F]
            
        return self.fc(self.drop(torch.cat(feats, 1)))