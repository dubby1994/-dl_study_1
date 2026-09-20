# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

CHANNELS = ["main_main", "main_carousel", "car1_carousel", "car2_carousel", "car3_carousel"]
K_RRF = 60
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 特征列定义 ----------
FEAT_COLS = []
for ch in CHANNELS:
    FEAT_COLS += [f"{ch}_mask", f"{ch}_rrf", f"{ch}_logidx", f"{ch}_idx_norm"]
    if f"{ch}_dist" in pd.read_parquet("samples_train.parquet").columns:
        FEAT_COLS.append(f"{ch}_dist")
FEAT_COLS += ["num_channels", "rrf_sum", "idx_min", "idx_mean", "main_cooccur"]


# ---------- 数据集 ----------
class RecallDataset(Dataset):
    def __init__(self, df):
        self.qids = df["query_item_id"].values
        self.x = torch.tensor(df[FEAT_COLS].values, dtype=torch.float32)
        self.y = torch.tensor(df["label"].values, dtype=torch.float32)
        # idx 原值，供可学习 RRF 模型使用
        idx_cols = [f"{ch}_idx" for ch in CHANNELS]
        mask = df[[f"{ch}_mask" for ch in CHANNELS]].values
        idx = df[idx_cols].fillna(0).values
        self.raw_idx = torch.tensor(idx, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.raw_idx[i], self.mask[i], self.y[i], self.qids[i]


# ---------- 模型 1：可学习 RRF（基线） ----------
class LearnableRRF(nn.Module):
    """score = Σ_i softplus(w_i) * mask_i / (softplus(k_i) + idx_i)"""

    def __init__(self, n_channels=5, k_init=60.0):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_channels))  # softplus(0)≈0.69
        self.k = nn.Parameter(torch.full((n_channels,),
                                         float(np.log(np.expm1(k_init)))))  # 反变换，使初值=60

    def forward(self, x, raw_idx, mask):
        w = torch.softplus(self.w)
        k = torch.softplus(self.k)
        score = (mask * w / (k + raw_idx)).sum(dim=1)
        return score


# ---------- 模型 2：MLP ----------
class FusionMLP(nn.Module):
    def __init__(self, in_dim, hidden=(64, 32), dropout=0.2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, raw_idx=None, mask=None):
        return self.net(x).squeeze(-1)


# ---------- Pairwise 损失（按 query 组内采样正负对） ----------
def pairwise_margin_loss(scores, labels, qids, margin=1.0, max_pairs=4096):
    pos_scores, neg_scores = [], []
    for qid in np.unique(qids):
        m = qids == qid
        s, y = scores[m], labels[m]
        p, n = s[y > 0.5], s[y < 0.5]
        if len(p) == 0 or len(n) == 0:
            continue
        # 笛卡尔积采样配对
        pi = torch.randint(len(p), (min(len(p) * len(n), max_pairs),), device=s.device)
        ni = torch.randint(len(n), (len(pi),), device=s.device)
        pos_scores.append(p[pi])
        neg_scores.append(n[ni])
    if not pos_scores:
        return None
    pos = torch.cat(pos_scores)
    neg = torch.cat(neg_scores)
    return torch.relu(margin - pos + neg).mean()


# ---------- 评估：Recall@K / NDCG@K（按 query 分组排序） ----------
@torch.no_grad()
def evaluate(model, df, ks=(50, 200, 500)):
    model.eval()
    ds = RecallDataset(df)
    dl = DataLoader(ds, batch_size=4096)
    all_scores = []
    for x, raw_idx, mask, _, _ in dl:
        s = model(x.to(DEVICE), raw_idx.to(DEVICE), mask.to(DEVICE))
        all_scores.append(s.cpu())
    df = df.copy()
    df["score"] = torch.cat(all_scores).numpy()

    recalls = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}
    for qid, g in df.groupby("query_item_id"):
        if g.label.sum() == 0:
            continue
        g = g.sort_values("score", ascending=False)
        ranked_labels = g.label.values
        total_pos = ranked_labels.sum()
        for k in ks:
            top = ranked_labels[:k]
            recalls[k].append(top.sum() / total_pos)
            dcg = (top / np.log2(np.arange(2, len(top) + 2))).sum()
            ideal = np.sort(ranked_labels)[::-1][:k]
            idcg = (ideal / np.log2(np.arange(2, len(ideal) + 2))).sum()
            ndcgs[k].append(dcg / idcg if idcg > 0 else 0)
    return {f"Recall@{k}": np.mean(v) for k, v in recalls.items()} | \
        {f"NDCG@{k}": np.mean(v) for k, v in ndcgs.items()}


# ---------- 训练主流程 ----------
def main(model_type="mlp", epochs=30, lr=1e-3, batch_size=2048):
    df = pd.read_parquet("samples_train.parquet")

    # 按 query 粒度切分，防止同一 query 的样本泄漏到训练和验证两边
    qids = df["query_item_id"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(qids)
    n_val = int(len(qids) * 0.15)
    val_q, train_q = qids[:n_val], qids[n_val:]
    train_df = df[df.query_item_id.isin(train_q)]
    val_df = df[df.query_item_id.isin(val_q)]
    print(f"train: {len(train_df)}, val: {len(val_df)}")

    if model_type == "rrf":
        model = LearnableRRF(n_channels=len(CHANNELS)).to(DEVICE)
    else:
        model = FusionMLP(in_dim=len(FEAT_COLS)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    train_dl = DataLoader(RecallDataset(train_df), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss, n_batch = 0.0, 0
        for x, raw_idx, mask, y, qid in train_dl:
            x, raw_idx, mask, y = (t.to(DEVICE) for t in (x, raw_idx, mask, y))
            scores = model(x, raw_idx, mask)
            loss = pairwise_margin_loss(scores, y, np.array(qid))
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batch += 1

        if (epoch + 1) % 5 == 0:
            metrics = evaluate(model, val_df)
            print(f"epoch {epoch + 1:3d} | loss {total_loss / max(n_batch, 1):.4f} | "
                  + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    torch.save(model.state_dict(), f"fusion_{model_type}.pt")

    # 打印学到的 RRF 参数，方便和线上直觉对照
    if model_type == "rrf":
        w = torch.softplus(model.w).detach().cpu().numpy()
        k = torch.softplus(model.k).detach().cpu().numpy()
        for ch, wi, ki in zip(CHANNELS, w, k):
            print(f"  {ch:15s} w={wi:.3f}  k={ki:.1f}")


if __name__ == "__main__":
    import sys

    main()
