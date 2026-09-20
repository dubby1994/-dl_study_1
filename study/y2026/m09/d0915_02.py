# ============================================================
# Recall Fusion MLP - Complete Demo
#
# 场景：
#   5路粗召回
#       1. 主主
#       2. 主轮
#       3. 轮1轮
#       4. 轮2轮
#       5. 轮3轮
#
# 输入：
#   [rank1, rank2, rank3, rank4, rank5]
#
# Teacher：
#   后置精排模型 score
#
# Training：
#   Pairwise RankNet + teacher score difference weight
#
# Evaluation：
#   NDCG@K
#   RFF baseline vs MLP
# ============================================================

import math
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


# ============================================================
# 0. Config
# ============================================================

SEED = 42

NUM_ROUTES = 5

RFF_K = 60.0

INPUT_DIM = 20

BATCH_SIZE = 256

EPOCHS = 30

LR = 1e-3

WEIGHT_DECAY = 1e-5

MAX_PAIRS_PER_QUERY = 500

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 1. Random Seed
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

print("Device:", DEVICE)


# ============================================================
# 2. Feature Engineering
# ============================================================

def rank_to_features(ranks):
    """
    输入：
        ranks = [
            rank_main_main,
            rank_main_carousel,
            rank_carousel_1,
            rank_carousel_2,
            rank_carousel_3
        ]

    None / 0 / -1：
        表示没有召回

    每一路 3 个 feature：

        is_recalled
        log(1 + rank)
        RFF

    5路：
        5 * 3 = 15

    Global：

        recall_cnt
        top1_cnt
        top10_cnt
        top50_cnt
        top100_cnt

    共：
        15 + 5 = 20维
    """

    assert len(ranks) == NUM_ROUTES

    features = []

    # --------------------------------------------------------
    # Per-route features
    # --------------------------------------------------------

    for rank in ranks:

        if rank is None or rank <= 0:

            is_recalled = 0.0
            log_rank = 0.0
            rff = 0.0

        else:

            is_recalled = 1.0

            log_rank = math.log1p(rank)

            rff = 1.0 / (RFF_K + rank)

        features.extend([
            is_recalled,
            log_rank,
            rff
        ])

    # --------------------------------------------------------
    # Global features
    # --------------------------------------------------------

    valid_ranks = [
        r for r in ranks
        if r is not None and r > 0
    ]

    recall_cnt = len(valid_ranks)

    top1_cnt = sum(
        r <= 1 for r in valid_ranks
    )

    top10_cnt = sum(
        r <= 10 for r in valid_ranks
    )

    top50_cnt = sum(
        r <= 50 for r in valid_ranks
    )

    top100_cnt = sum(
        r <= 100 for r in valid_ranks
    )

    features.extend([
        float(recall_cnt),
        float(top1_cnt),
        float(top10_cnt),
        float(top50_cnt),
        float(top100_cnt),
    ])

    assert len(features) == INPUT_DIM

    return np.asarray(
        features,
        dtype=np.float32
    )


# ============================================================
# 3. Demo Dataset
# ============================================================

def generate_demo_data(
    num_queries=300,
    candidates_per_query=50,
):
    """
    构造模拟数据。

    注意：

    这里的 teacher score 并不是简单 RFF。

    它模拟：
        精排模型综合考虑了图片相似度、
        多路证据、路由权重等因素。

    因此 MLP 有机会学习出：
        主主 > 主轮 > 轮播轮
        多路同时召回有额外价值
        rank越靠前价值越高
    """

    data = []

    # --------------------------------------------------------
    # 真实的 teacher 生成逻辑
    #
    # 注意：
    # 这只是为了生成 demo 数据。
    #
    # 实际项目中这里应该直接替换成：
    #     你的精排模型 score
    # --------------------------------------------------------

    route_weights = np.array([
        2.5,   # 主主
        1.5,   # 主轮
        1.0,   # 轮1
        0.8,   # 轮2
        0.7,   # 轮3
    ])

    for q in range(num_queries):

        query_id = f"query_{q}"

        for sku_idx in range(candidates_per_query):

            # ------------------------------------------------
            # 随机产生5路 rank
            # ------------------------------------------------

            ranks = []

            for route in range(NUM_ROUTES):

                # 35%概率该路没有召回
                if random.random() < 0.35:

                    ranks.append(None)

                else:

                    # rank更偏向前排
                    rank = int(
                        np.random.exponential(
                            scale=80
                        )
                    ) + 1

                    rank = min(
                        rank,
                        2000
                    )

                    ranks.append(rank)

            # 至少一路召回
            if all(
                r is None
                for r in ranks
            ):
                ranks[
                    random.randint(0, NUM_ROUTES - 1)
                ] = random.randint(1, 100)

            # ------------------------------------------------
            # 模拟 teacher score
            # ------------------------------------------------

            route_score = 0.0

            for i, rank in enumerate(ranks):

                if rank is None:
                    continue

                # 比RFF衰减更明显一点
                similarity = 1.0 / math.sqrt(
                    1.0 + rank
                )

                route_score += (
                    route_weights[i]
                    * similarity
                )

            # ------------------------------------------------
            # 多路召回协同效应
            # ------------------------------------------------

            valid_count = sum(
                r is not None
                for r in ranks
            )

            interaction_bonus = 0.0

            if valid_count >= 2:
                interaction_bonus += 0.5

            if valid_count >= 3:
                interaction_bonus += 0.8

            if valid_count >= 4:
                interaction_bonus += 1.0

            # ------------------------------------------------
            # 加一点随机噪声
            # ------------------------------------------------

            noise = np.random.normal(
                loc=0.0,
                scale=0.15
            )

            teacher_score = (
                route_score
                + interaction_bonus
                + noise
            )

            data.append({
                "query_id": query_id,
                "sku_id": f"{query_id}_sku_{sku_idx}",
                "ranks": ranks,
                "teacher_score": float(
                    teacher_score
                )
            })

    return data


# ============================================================
# 4. Build Pairwise Dataset
# ============================================================

class PairwiseDataset(Dataset):

    def __init__(
        self,
        data,
        max_pairs_per_query=500
    ):

        self.pairs = []

        # ----------------------------------------------------
        # query 分组
        # ----------------------------------------------------

        query_groups = defaultdict(list)

        for item in data:

            query_groups[
                item["query_id"]
            ].append(item)

        # ----------------------------------------------------
        # 每个 query 构造 pair
        # ----------------------------------------------------

        for query_id, items in query_groups.items():

            if len(items) < 2:
                continue

            # ------------------------------------------------
            # 按 teacher score 排序
            # ------------------------------------------------

            items = sorted(
                items,
                key=lambda x: x["teacher_score"],
                reverse=True
            )

            n = len(items)

            # ------------------------------------------------
            # 直接构造所有pair
            #
            # Demo数据只有50个候选，
            # 真实数据不要这样做。
            # ------------------------------------------------

            candidate_pairs = []

            for i in range(n):

                for j in range(i + 1, n):

                    pos = items[i]
                    neg = items[j]

                    teacher_diff = (
                        pos["teacher_score"]
                        - neg["teacher_score"]
                    )

                    # teacher score 完全相同时没有监督意义
                    if teacher_diff <= 1e-6:
                        continue

                    candidate_pairs.append(
                        (
                            pos,
                            neg,
                            teacher_diff
                        )
                    )

            # ------------------------------------------------
            # 控制每个 query pair数量
            # ------------------------------------------------

            if len(candidate_pairs) > max_pairs_per_query:

                candidate_pairs = random.sample(
                    candidate_pairs,
                    max_pairs_per_query
                )

            self.pairs.extend(
                candidate_pairs
            )

        print(
            f"Pairwise dataset: "
            f"{len(self.pairs):,} pairs"
        )

    def __len__(self):

        return len(self.pairs)

    def __getitem__(self, idx):

        pos, neg, teacher_diff = self.pairs[idx]

        pos_feature = rank_to_features(
            pos["ranks"]
        )

        neg_feature = rank_to_features(
            neg["ranks"]
        )

        return (
            torch.from_numpy(pos_feature),
            torch.from_numpy(neg_feature),
            torch.tensor(
                teacher_diff,
                dtype=torch.float32
            )
        )


# ============================================================
# 5. MLP Model
# ============================================================

class RecallFusionMLP(nn.Module):

    def __init__(
        self,
        input_dim=INPUT_DIM
    ):

        super().__init__()

        self.mlp = nn.Sequential(

            nn.Linear(
                input_dim,
                64
            ),

            nn.LayerNorm(64),

            nn.ReLU(),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                16
            ),

            nn.ReLU(),

            nn.Linear(
                16,
                1
            )
        )

    def forward(self, x):

        return self.mlp(x).squeeze(-1)


# ============================================================
# 6. Weighted RankNet Loss
# ============================================================

class WeightedRankNetLoss(nn.Module):

    def __init__(
        self,
        min_weight=0.1,
        max_weight=5.0
    ):

        super().__init__()

        self.min_weight = min_weight
        self.max_weight = max_weight

    def forward(
        self,
        pos_score,
        neg_score,
        teacher_diff
    ):

        # ----------------------------------------------------
        # MLP希望：
        #
        #     pos_score > neg_score
        #
        # diff越大越好
        # ----------------------------------------------------

        score_diff = (
            pos_score
            - neg_score
        )

        # ----------------------------------------------------
        # teacher差距作为weight
        # ----------------------------------------------------

        weight = torch.clamp(
            teacher_diff,
            min=self.min_weight,
            max=self.max_weight
        )

        # ----------------------------------------------------
        # RankNet
        #
        # log(1 + exp(-score_diff))
        #
        # softplus 数值更加稳定
        # ----------------------------------------------------

        pair_loss = F.softplus(
            -score_diff
        )

        weighted_loss = (
            weight * pair_loss
        )

        return weighted_loss.mean()


# ============================================================
# 7. Train One Epoch
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device
):

    model.train()

    total_loss = 0.0

    total_count = 0

    for (
        pos_x,
        neg_x,
        teacher_diff
    ) in dataloader:

        pos_x = pos_x.to(device)

        neg_x = neg_x.to(device)

        teacher_diff = teacher_diff.to(device)

        # ----------------------------------------------------
        # MLP score
        # ----------------------------------------------------

        pos_score = model(
            pos_x
        )

        neg_score = model(
            neg_x
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            pos_score,
            neg_score,
            teacher_diff
        )

        # ----------------------------------------------------
        # backward
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        batch_size = pos_x.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_count += batch_size

    return total_loss / max(
        total_count,
        1
    )


# ============================================================
# 8. NDCG
# ============================================================

def dcg(relevances):

    score = 0.0

    for i, rel in enumerate(relevances):

        rank = i + 1

        score += (
            (2 ** rel - 1)
            / math.log2(rank + 1)
        )

    return score


def ndcg_at_k(
    labels,
    scores,
    k=20
):

    labels = np.asarray(
        labels
    )

    scores = np.asarray(
        scores
    )

    order = np.argsort(
        -scores
    )

    pred_labels = labels[
        order[:k]
    ]

    ideal_labels = np.sort(
        labels
    )[::-1][:k]

    dcg_value = dcg(
        pred_labels
    )

    idcg_value = dcg(
        ideal_labels
    )

    if idcg_value <= 0:

        return 0.0

    return (
        dcg_value
        / idcg_value
    )


# ============================================================
# 9. Evaluate MLP
# ============================================================

@torch.no_grad()
def evaluate_mlp(
    model,
    data,
    device,
    k=20
):

    model.eval()

    query_groups = defaultdict(list)

    for item in data:

        query_groups[
            item["query_id"]
        ].append(item)

    all_ndcg = []

    for query_id, items in query_groups.items():

        features = np.stack([
            rank_to_features(
                x["ranks"]
            )
            for x in items
        ])

        x = torch.from_numpy(
            features
        ).to(device)

        scores = (
            model(x)
            .cpu()
            .numpy()
        )

        labels = [
            x["teacher_score"]
            for x in items
        ]

        value = ndcg_at_k(
            labels,
            scores,
            k=k
        )

        all_ndcg.append(value)

    return float(
        np.mean(all_ndcg)
    )


# ============================================================
# 10. Evaluate RFF Baseline
# ============================================================

def calculate_rff_score(ranks):

    score = 0.0

    for rank in ranks:

        if rank is None or rank <= 0:
            continue

        score += (
            1.0
            / (RFF_K + rank)
        )

    return score


def evaluate_rff(
    data,
    k=20
):

    query_groups = defaultdict(list)

    for item in data:

        query_groups[
            item["query_id"]
        ].append(item)

    all_ndcg = []

    for query_id, items in query_groups.items():

        scores = [
            calculate_rff_score(
                x["ranks"]
            )
            for x in items
        ]

        labels = [
            x["teacher_score"]
            for x in items
        ]

        value = ndcg_at_k(
            labels,
            scores,
            k=k
        )

        all_ndcg.append(value)

    return float(
        np.mean(all_ndcg)
    )


# ============================================================
# 11. Show Ranking Example
# ============================================================

@torch.no_grad()
def show_example(
    model,
    data,
    query_id,
    device,
    topk=10
):

    model.eval()

    items = [
        x
        for x in data
        if x["query_id"] == query_id
    ]

    features = np.stack([
        rank_to_features(
            x["ranks"]
        )
        for x in items
    ])

    x = torch.from_numpy(
        features
    ).to(device)

    mlp_scores = (
        model(x)
        .cpu()
        .numpy()
    )

    result = []

    for item, mlp_score in zip(
        items,
        mlp_scores
    ):

        result.append({

            "sku_id":
                item["sku_id"],

            "ranks":
                item["ranks"],

            "teacher":
                item["teacher_score"],

            "rff":
                calculate_rff_score(
                    item["ranks"]
                ),

            "mlp":
                float(mlp_score)
        })

    print("\n")
    print("=" * 100)

    print(
        f"Query: {query_id}"
    )

    print(
        "Teacher Ranking"
    )

    print("=" * 100)

    teacher_sorted = sorted(
        result,
        key=lambda x: x["teacher"],
        reverse=True
    )

    for i, x in enumerate(
        teacher_sorted[:topk],
        1
    ):

        print(
            f"{i:02d} "
            f"{x['sku_id']:25s} "
            f"ranks={str(x['ranks']):35s} "
            f"teacher={x['teacher']:.4f}"
        )

    print("\n")

    print(
        "RFF Ranking"
    )

    print("=" * 100)

    rff_sorted = sorted(
        result,
        key=lambda x: x["rff"],
        reverse=True
    )

    for i, x in enumerate(
        rff_sorted[:topk],
        1
    ):

        print(
            f"{i:02d} "
            f"{x['sku_id']:25s} "
            f"ranks={str(x['ranks']):35s} "
            f"rff={x['rff']:.6f}"
        )

    print("\n")

    print(
        "MLP Ranking"
    )

    print("=" * 100)

    mlp_sorted = sorted(
        result,
        key=lambda x: x["mlp"],
        reverse=True
    )

    for i, x in enumerate(
        mlp_sorted[:topk],
        1
    ):

        print(
            f"{i:02d} "
            f"{x['sku_id']:25s} "
            f"ranks={str(x['ranks']):35s} "
            f"mlp={x['mlp']:.6f} "
            f"teacher={x['teacher']:.4f}"
        )


# ============================================================
# 12. Main
# ============================================================

def main():

    # ========================================================
    # Generate Data
    # ========================================================

    print("\nGenerating demo data...")

    data = generate_demo_data(
        num_queries=300,
        candidates_per_query=50
    )

    print(
        f"Total samples: {len(data):,}"
    )

    # ========================================================
    # Query-level Train / Valid Split
    # ========================================================

    query_ids = sorted(
        set(
            x["query_id"]
            for x in data
        )
    )

    random.shuffle(
        query_ids
    )

    split_idx = int(
        len(query_ids) * 0.8
    )

    train_query_ids = set(
        query_ids[:split_idx]
    )

    valid_query_ids = set(
        query_ids[split_idx:]
    )

    train_data = [
        x
        for x in data
        if x["query_id"]
        in train_query_ids
    ]

    valid_data = [
        x
        for x in data
        if x["query_id"]
        in valid_query_ids
    ]

    print(
        f"Train queries: "
        f"{len(train_query_ids)}"
    )

    print(
        f"Valid queries: "
        f"{len(valid_query_ids)}"
    )

    print(
        f"Train samples: "
        f"{len(train_data):,}"
    )

    print(
        f"Valid samples: "
        f"{len(valid_data):,}"
    )

    # ========================================================
    # RFF Baseline
    # ========================================================

    baseline_ndcg = evaluate_rff(
        valid_data,
        k=20
    )

    print(
        f"\nRFF baseline "
        f"NDCG@20 = "
        f"{baseline_ndcg:.6f}"
    )

    # ========================================================
    # Pairwise Dataset
    # ========================================================

    train_dataset = PairwiseDataset(
        train_data,
        max_pairs_per_query=
            MAX_PAIRS_PER_QUERY
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    # ========================================================
    # Model
    # ========================================================

    model = RecallFusionMLP(
        input_dim=INPUT_DIM
    ).to(DEVICE)

    print("\nModel:")
    print(model)

    # ========================================================
    # Loss
    # ========================================================

    criterion = WeightedRankNetLoss(
        min_weight=0.1,
        max_weight=5.0
    )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # Training
    # ========================================================

    best_ndcg = -1.0

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            DEVICE
        )

        valid_ndcg = evaluate_mlp(
            model,
            valid_data,
            DEVICE,
            k=20
        )

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.6f} | "
            f"NDCG@20={valid_ndcg:.6f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if valid_ndcg > best_ndcg:

            best_ndcg = valid_ndcg

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "input_dim":
                        INPUT_DIM,

                    "best_ndcg":
                        best_ndcg
                },
                "recall_fusion_mlp.pt"
            )

    # ========================================================
    # Final Result
    # ========================================================

    print("\n")
    print("=" * 70)

    print(
        f"RFF NDCG@20 : "
        f"{baseline_ndcg:.6f}"
    )

    print(
        f"MLP NDCG@20 : "
        f"{best_ndcg:.6f}"
    )

    print(
        f"Improvement  : "
        f"{best_ndcg - baseline_ndcg:+.6f}"
    )

    print("=" * 70)

    # ========================================================
    # Show Example
    # ========================================================

    example_query = sorted(
        valid_query_ids
    )[0]

    show_example(
        model,
        valid_data,
        example_query,
        DEVICE,
        topk=10
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()