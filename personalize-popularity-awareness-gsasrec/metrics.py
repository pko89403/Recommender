import torch


def compuate_metrics(original_logits, labels, num_items, k=5, eps=1e-10):
    # 시그모이드를 이용해 확률로 변환 (logits의 크기는 (batch, seq_length-1, n_candidates))
    # 실제 레이블 분포
    original_logits = original_logits
    probabilities = torch.sigmoid(original_logits)

    mask = (
        (labels != num_items + 1).float().view(labels.size(0), 1)
    )  # (batch, 1)
    
    # 1. Top-1 Accuracy: 각 위치에서 가장 높은 확률을 가진 인덱스가 0이면 정답.
    pred_indices = torch.argmax(probabilities, dim=-1)  # (batch, seq_length-1)
    top1_correct = (pred_indices == 0).float() * mask  # mask를 곱해 pad 영역은 무시
    top1_accuracy = top1_correct.sum() / mask.sum()

    # 2. Recall@K (Hit Rate@K): 상위 K 예측 중 정답(인덱스 0)이 존재하는지 확인.
    topk_values, topk_indices = torch.topk(
        probabilities, k, dim=-1
    )  # (batch, seq_length-1, k)
    # topk_indices에 0이 하나라도 포함되면 hit로 간주
    hits = (topk_indices == 0).any(dim=-1).float() * mask
    recall_at_k = hits.sum() / mask.sum()

    # 3. NDCG@K: 정답(항상 인덱스 0)의 순위를 이용해 DCG 계산
    # 양성 샘플의 순위는, 해당 위치의 로짓과 비교해 몇 개의 후보가 더 높은 점수를 가지는지에 따라 결정
    # rank = (더 높은 로짓의 개수) + 1
    ranks = (original_logits > original_logits[:, :, 0:1]).sum(
        dim=-1
    ) + 1  # (batch, seq_length-1)
    # NDCG 계산: 만약 rank가 k 이하이면 DCG = 1 / log2(rank + 1), 아니면 0
    ndcg = torch.where(
        ranks <= k,
        1.0 / torch.log2(ranks.to(torch.float64) + 1),
        torch.zeros_like(ranks, dtype=torch.float64),
    )
    ndcg_at_k = (ndcg * mask).sum() / mask.sum()

    return top1_accuracy.item(), recall_at_k.item(), ndcg_at_k.item()
