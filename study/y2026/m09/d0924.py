import torch
import torch.nn as nn
from torch.nn import functional as F

# --- 超参数设置 ---
batch_size = 32  # 批处理大小
block_size = 16  # 上下文窗口大小（自回归的最大长度）
max_iters = 1000  # 训练迭代次数
learning_rate = 1e-3
n_embd = 128  # 嵌入维度
n_head = 8  # 注意力头数
n_layer = 16  # Transformer 块数
dropout = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 设置随机种子以保证结果可复现
torch.manual_seed(1337)

# --- 1. 数据准备 ---
# 使用一段简单的文本作为训练语料
text = "床前明月光，疑是地上霜。" \
       "举头望明月，低头思故乡。" \
       "春眠不觉晓，处处闻啼鸟。" \
       "夜来风雨声，花落知多少。" \
       "白日依山尽，黄河入海流。" \
       "欲穷千里目，更上一层楼。" \
       "红豆生南国，春来发几枝。" \
       "愿君多采撷，此物最相思。" \
       "空山不见人，但闻人语响。" \
       "返景入深林，复照青苔上。" \
       "千山鸟飞绝，万径人踪灭。" \
       "孤舟蓑笠翁，独钓寒江雪。" \
       "松下问童子，言师采药去。" \
       "只在此山中，云深不知处。" \
       "锄禾日当午，汗滴禾下土。" \
       "谁知盘中餐，粒粒皆辛苦。" \
       "迟日江山丽，春风花草香。" \
       "泥融飞燕子，沙暖睡鸳鸯。" \
       "月黑雁飞高，单于夜遁逃。" \
       "欲将轻骑逐，大雪满弓刀。" \
       "千山鸟飞绝，万径人踪灭。" \
       "孤舟蓑笠翁，独钓寒江雪。" \
       "白日依山尽，黄河入海流。" \
       "欲穷千里目，更上一层楼。" \
       "红豆生南国，春来发几枝。" \
       "愿君多采撷，此物最相思。" \
       "空山不见人，但闻人语响。" \
       "返景入深林，复照青苔上。" \
       "松下问童子，言师采药去。" \
       "只在此山中，云深不知处。" \
       "锄禾日当午，汗滴禾下土。" \
       "谁知盘中餐，粒粒皆辛苦。" \
       "迟日江山丽，春风花草香。" \
       "泥融飞燕子，沙暖睡鸳鸯。" \
       "月黑雁飞高，单于夜遁逃。" \
       "欲将轻骑逐，大雪满弓刀。" \
 \
# 获取所有唯一的字符，构建词表
chars = sorted(list(set(text)))
vocab_size = len(chars)

# 建立字符到整数的映射
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 转换为张量
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    """生成用于训练的小批次数据 (x: 输入, y: 标签)"""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# --- 2. 模型架构组件 ---

class Head(nn.Module):
    """ 单头自注意力机制 """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # 自回归掩码：确保模型只能看到当前位置及之前的信息（防止作弊）
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # 计算注意力得分 (Query 点积 Key)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, T)
        # 应用自回归掩码，将未来时刻的权重设为负无穷
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # 对 Value 进行加权求和
        v = self.value(x)
        out = wei @ v  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """ 多头注意力机制 """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class Block(nn.Module):
    """ Transformer 块：注意力 + 前馈神经网络 """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # 带有残差连接和层归一化的结构
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class NanoGPT(nn.Module):
    """ 完整的最小 GPT 模型 """

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # 1. 获取 Token 嵌入和位置嵌入
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)

        # 2. 通过 Transformer 块
        x = self.blocks(x)
        x = self.ln_f(x)

        # 3. 映射到词表大小，预测下一个 Token
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """ 自回归生成文本的核心逻辑 """
        for _ in range(max_new_tokens):
            # 裁剪输入，确保不超过上下文窗口大小 (block_size)
            idx_cond = idx[:, -block_size:]
            # 前向传播获取预测结果
            logits, _ = self(idx_cond)
            # 仅关注最后一个时间步的输出
            logits = logits[:, -1, :]  # (B, C)
            # 转化为概率分布
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # 从中采样下一个字符
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # 拼接回输入序列，继续下一步生成
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# --- 3. 训练与推理流程 ---

if __name__ == '__main__':
    model = NanoGPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e3:.1f} KB")
    print("开始训练...\n")

    # 简单训练循环
    for iter in range(max_iters):
        xb, yb = get_batch('train')

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True) if hasattr(optimizer, 'zero_grad') else optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 200 == 0:
            print(f"迭代步数 {iter:4d} | 损失函数值: {loss.item():.4f}")

    print("\n训练完成！开始自回归文本生成推理：\n" + "-" * 40)

    # 推理：从一个全零的空白字符（或指定提示词）开始生成
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_indices = model.generate(context, max_new_tokens=300)[0].tolist()
    print(decode(generated_indices))
