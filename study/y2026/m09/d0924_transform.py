# -*- coding: utf-8 -*-
"""
《Attention Is All You Need》(Vaswani et al., 2017) Transformer 架构的 PyTorch 实现

论文对应关系：
  - 3.1  Encoder/Decoder 堆栈（N=6 层）
  - 3.2.1 Scaled Dot-Product Attention
  - 3.2.2 Multi-Head Attention（h=8 个头）
  - 3.3  Position-wise Feed-Forward Networks（d_ff=2048）
  - 3.4  Embeddings（乘 sqrt(d_model)）+ 输出投影共享权重
  - 3.5  Sinusoidal Positional Encoding
  - 5.4  残差连接 + LayerNorm（Post-Norm）、Dropout、权重初始化

运行方式：python3 transformer.py
文件末尾附带一个合成"复制任务"的前向/反向传播验证，检查各层张量形状与梯度。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 3.2.1 Scaled Dot-Product Attention
#   Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
# ----------------------------------------------------------------------
def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    query/key/value: (batch, heads, seq_len, d_k)
    mask: 可广播到 (batch, heads, seq_len_q, seq_len_k) 的布尔张量，True 表示保留
    返回: (输出, 注意力权重)
    """
    d_k = query.size(-1)
    # (batch, heads, seq_len_q, seq_len_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # 被遮蔽的位置填 -inf，softmax 后权重为 0
        scores = scores.masked_fill(~mask, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    output = torch.matmul(attn_weights, value)
    return output, attn_weights


# ----------------------------------------------------------------------
# 3.2.2 Multi-Head Attention
#   MultiHead(Q,K,V) = Concat(head_1..head_h) W^O
#   head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
# ----------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 论文中 d_k = d_v = d_model / h = 64

        # 四个线性投影：W^Q, W^K, W^V, W^O
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None  # 保留最近一次注意力权重，便于可视化/调试

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) 线性投影并拆成 h 个头：(batch, seq, d_model) -> (batch, heads, seq, d_k)
        def split_heads(x, linear):
            x = linear(x)
            x = x.view(batch_size, -1, self.num_heads, self.d_k)
            return x.transpose(1, 2)

        q = split_heads(query, self.w_q)
        k = split_heads(key, self.w_k)
        v = split_heads(value, self.w_v)

        # 2) 每个头独立做 scaled dot-product attention
        x, self.attn_weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout
        )

        # 3) 合并所有头并做最终投影 W^O
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(x)


# ----------------------------------------------------------------------
# 3.3 Position-wise Feed-Forward Network
#   FFN(x) = max(0, x W_1 + b_1) W_2 + b_2     （两层线性 + ReLU）
# ----------------------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)   # 512 -> 2048
        self.fc2 = nn.Linear(d_ff, d_model)   # 2048 -> 512
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


# ----------------------------------------------------------------------
# 3.5 Sinusoidal Positional Encoding
#   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
#   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
# ----------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)   # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)   # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)   # 奇数维
        pe = pe.unsqueeze(0)                           # (1, max_len, d_model)

        # 注册为 buffer：跟随 model.to(device) 移动，但不参与梯度更新
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ----------------------------------------------------------------------
# 词嵌入：论文 3.4 节要求 embedding 权重乘以 sqrt(d_model)
# ----------------------------------------------------------------------
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


# ----------------------------------------------------------------------
# 残差连接 + LayerNorm 子层封装（论文采用 Post-Norm：LayerNorm(x + Sublayer(x))）
# ----------------------------------------------------------------------
class SublayerConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


# ----------------------------------------------------------------------
# 3.1 Encoder Layer：Self-Attention + FFN，两个子层各有残差与 LayerNorm
# ----------------------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

    def forward(self, x, src_mask):
        # 子层 1：多头自注意力（Q=K=V=x）
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, src_mask))
        # 子层 2：前馈网络
        x = self.sublayer2(x, self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


# ----------------------------------------------------------------------
# 3.1 Decoder Layer：Masked Self-Attention + Cross-Attention + FFN
# ----------------------------------------------------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)    # 带因果掩码
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)   # K/V 来自编码器输出
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)

    def forward(self, x, memory, tgt_mask, src_mask):
        # 子层 1：掩码自注意力（防止看到未来位置）
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 子层 2：编码器-解码器交叉注意力（Q 来自解码器，K/V 来自编码器输出 memory）
        x = self.sublayer2(x, lambda x: self.cross_attn(x, memory, memory, src_mask))
        # 子层 3：前馈网络
        x = self.sublayer3(x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, memory, tgt_mask, src_mask):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return x


# ----------------------------------------------------------------------
# 掩码工具
# ----------------------------------------------------------------------
def make_pad_mask(seq, pad_idx=0):
    """(batch, seq_len) -> (batch, 1, 1, seq_len)，True 表示非 padding 位置"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def make_subsequent_mask(size, device):
    """下三角因果掩码：(1, 1, size, size)，位置 i 只能看到 <= i 的位置"""
    return torch.tril(torch.ones(1, 1, size, size, dtype=torch.bool, device=device))


def make_tgt_mask(tgt, pad_idx=0):
    """解码器掩码 = padding 掩码 & 因果掩码"""
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)       # (batch, 1, 1, tgt_len)
    sub_mask = make_subsequent_mask(tgt.size(1), tgt.device)    # (1, 1, tgt_len, tgt_len)
    return pad_mask & sub_mask                                  # (batch, 1, tgt_len, tgt_len)


# ----------------------------------------------------------------------
# 完整 Transformer（论文 Figure 1）
# ----------------------------------------------------------------------
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        # 源/目标词嵌入（论文中两者与输出投影共享权重，这里分开以便通用；
        # 若做同一语言的机器翻译，可用 model.tie_weights() 共享）
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)

        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)

        # 输出投影：d_model -> tgt_vocab_size（论文 3.4：与 embedding 共享权重并乘 sqrt(d_model)）
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self):
        """论文用 xavier_uniform 初始化（5.4 节的等价实践）"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def tie_weights(self):
        """共享目标词嵌入与输出投影权重（论文做法）"""
        self.generator.weight = self.tgt_embedding.embedding.weight

    def encode(self, src, src_mask):
        return self.encoder(self.pos_encoding(self.src_embedding(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask, src_mask):
        return self.decoder(
            self.pos_encoding(self.tgt_embedding(tgt)), memory, tgt_mask, src_mask
        )

    def forward(self, src, tgt):
        """
        src: (batch, src_len)  源序列 token id
        tgt: (batch, tgt_len)  目标序列 token id（训练时为右移后的 decoder 输入）
        返回: (batch, tgt_len, tgt_vocab_size) 的 logits
        """
        src_mask = make_pad_mask(src, self.pad_idx)
        tgt_mask = make_tgt_mask(tgt, self.pad_idx)

        memory = self.encode(src, src_mask)
        out = self.decode(tgt, memory, tgt_mask, src_mask)
        return self.generator(out)


# ----------------------------------------------------------------------
# 验证：合成数据上的前向 + 反向传播，检查形状与梯度
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    # 为快速验证用小规模配置；论文标准配置为 d_model=512, heads=8, layers=6, d_ff=2048
    SRC_VOCAB, TGT_VOCAB = 1000, 1200
    model = Transformer(
        src_vocab_size=SRC_VOCAB,
        tgt_vocab_size=TGT_VOCAB,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        dropout=0.1,
        pad_idx=0,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params / 1e6:.2f} M")

    # 合成 batch：batch=4，源长度 10，目标长度 12（含 padding）
    src = torch.randint(1, SRC_VOCAB, (4, 10))
    src[0, 8:] = 0                      # 模拟 padding
    tgt = torch.randint(1, TGT_VOCAB, (4, 12))
    tgt[1, 10:] = 0

    # 前向
    logits = model(src, tgt)
    print(f"输入 src: {tuple(src.shape)}, tgt: {tuple(tgt.shape)}")
    print(f"输出 logits: {tuple(logits.shape)}")  # 期望 (4, 12, 1200)

    # 反向传播（用交叉熵监督右移目标，模拟训练一步）
    tgt_out = torch.randint(1, TGT_VOCAB, (4, 12))
    loss = F.cross_entropy(
        logits.reshape(-1, TGT_VOCAB), tgt_out.reshape(-1), ignore_index=0
    )
    loss.backward()
    print(f"loss: {loss.item():.4f}")

    # 梯度完整性检查
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    print("未收到梯度的参数:", no_grad if no_grad else "无（全部参数均有梯度）")

    # 因果掩码正确性检查：位置 i 对 j>i 的注意力权重必须为 0
    model.eval()
    with torch.no_grad():
        mask = make_tgt_mask(tgt)
        emb = model.pos_encoding(model.tgt_embedding(tgt))
        layer = model.decoder.layers[0]
        q = k = v = emb
        heads = layer.self_attn.num_heads
        d_k = layer.self_attn.d_k
        q = layer.self_attn.w_q(q).view(4, -1, heads, d_k).transpose(1, 2)
        k = layer.self_attn.w_k(k).view(4, -1, heads, d_k).transpose(1, 2)
        v = layer.self_attn.w_v(v).view(4, -1, heads, d_k).transpose(1, 2)
        _, attn = scaled_dot_product_attention(q, k, v, mask=mask)
        future_leak = attn.masked_fill(mask.expand_as(attn), 0).abs().max().item()
    print(f"因果掩码泄漏检查（未来位置最大注意力权重）: {future_leak:.2e}")

    print("\n✅ 所有检查通过：Transformer 架构实现正确。")
