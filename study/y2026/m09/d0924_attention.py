import torch
import torch.nn.functional as F

def attention(Q, K, V):
    """
    Q, K, V: (batch, seq_len, d_k)
    返回: 注意力输出, 注意力权重
    """
    d_k = Q.size(-1)
    # 1. 计算相似度分数: Q @ K^T, 并缩放
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)   # (batch, seq_len, seq_len)
    # 2. softmax 归一化得到注意力权重
    weights = F.softmax(scores, dim=-1)
    # 3. 加权求和 V
    output = weights @ V                              # (batch, seq_len, d_k)
    return output, weights


# ---- 测试 ----
torch.manual_seed(0)
batch, seq_len, d_k = 2, 4, 8

Q = torch.randn(batch, seq_len, d_k)
K = torch.randn(batch, seq_len, d_k)
V = torch.randn(batch, seq_len, d_k)

out, w = attention(Q, K, V)
print("输出形状:", out.shape)        # (2, 4, 8)
print("权重形状:", w.shape)          # (2, 4, 4)
print("每行权重和:", w.sum(-1))      # 每行都约等于 1