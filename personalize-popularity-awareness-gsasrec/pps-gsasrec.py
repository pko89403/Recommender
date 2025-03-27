#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
import polars as pl
from torch.utils.data import DataLoader
import torch
import numpy as np
import wandb

from dataset import SequenceDataset, collate_with_inbatch_negatives
from model import GSASRec
from metrics import compuate_metrics


def load_and_prepare_metadata(track_name, character_name, ips_temperature, max_genres_per_track):
    tracks = pl.read_parquet(track_name)
    _ = pl.read_parquet(character_name)
    num_items = len(tracks)

    # IPS 계산을 위한 track_index 및 track_count 추출
    propensity_score = np.array(pl.Series(tracks.sort("track_index").select("track_index")).to_list())
    track_ips = propensity_score / propensity_score.sum() + 1e-6

    adjusted_ips = np.power(track_ips, ips_temperature)
    neg_sampling_prob = adjusted_ips / adjusted_ips.sum()

    tracks_selected_columns = tracks.select(
        ["track_index", "year_category", "domain", "genre_id_list"]
    )

    genre_mapping = defaultdict(lambda: 0)  # outlier 값은 0으로 설정
    domain_mapping = defaultdict(lambda: 0)  # outlier 값은 0으로 설정
    year_category_mapping = defaultdict(lambda: 0)  # outlier 값은 0으로 설정

    next_year_category_index = 1
    next_domain_index = 1
    next_genre_index = 1

    track_meta_dict = defaultdict(
        lambda: {
            "year_category": 0,
            "domain": 0,
            "genre_id_list": [0] * max_genres_per_track,
        }
    )

    for row in tracks_selected_columns.to_dicts():
        # None 값 처리 (year_category, domain, genre_id_list)
        year_category = (
            row["year_category"]
            if row["year_category"] is not None and row["year_category"] != ""
            else None
        )
        domain = (
            row["domain"] if row["domain"] is not None and row["domain"] != "" else None
        )
        genre_id_list = row["genre_id_list"] if row["genre_id_list"] is not None else []

        # year_category 매핑 추가 (None과 `'` 제외)
        if year_category not in year_category_mapping and year_category is not None:
            year_category_mapping[year_category] = next_year_category_index
            next_year_category_index += 1

        # domain 매핑 추가 (None과 `'` 제외)
        if domain not in domain_mapping and domain is not None:
            domain_mapping[domain] = next_domain_index
            next_domain_index += 1

        # genre_id_list 매핑 추가 및 변환 (None과 `'` 제외)
        genre_ids_mapped = []
        for genre in genre_id_list:
            if genre not in genre_mapping and genre is not None and genre != "":
                genre_mapping[genre] = next_genre_index
                next_genre_index += 1
            genre_ids_mapped.append(genre_mapping[genre])

        # genre_id_list 크기 고정 (최대 MAX_GENRES_PER_TRACK, 부족하면 패딩)
        genre_ids_mapped = genre_ids_mapped[:max_genres_per_track] + [0] * (
            max_genres_per_track - len(genre_ids_mapped)
        )

        # track_meta_dict 업데이트
        track_meta_dict[row["track_index"]] = {
            "year_category": year_category_mapping[year_category],
            "domain": domain_mapping[domain],
            "genre_id_list": genre_ids_mapped,
        }
        
        return (tracks, num_items, neg_sampling_prob, track_meta_dict, genre_mapping, domain_mapping, year_category_mapping)


def initialize_training(args):
    tracks, num_items, neg_sampling_prob, track_meta_dict, genre_mapping, domain_mapping, year_category_mapping = load_and_prepare_metadata(
        args.track_name, args.character_name, args.ips_temperature, args.max_genres_per_track
    )
    padding_value = num_items + 1
    
    train_dataset = SequenceDataset(
        args.dataset_name,
        track_meta_dict,
        len(tracks) + 1,
        args.sequence_length + 1,
        args.n_negatives,
        neg_sampling_prob,
        args.max_genres_per_track,
        args.pps_eps
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_with_inbatch_negatives(x, len(tracks) + 1, track_meta_dict),
        num_workers=args.num_workers,
    )


    wandb.init(
        project="popgsasrec", name="gsasrec"
    )

    model = GSASRec(
        num_items=num_items,
        padding_value=padding_value,
        embedding_id_dim=args.embedding_dim_id,
        genre_mapping=genre_mapping,
        embedding_genre_dim=args.embedding_dim_genre,
        domain_mapping=domain_mapping,
        embedding_domain_dim=args.embedding_dim_domain,
        year_category_mapping=year_category_mapping,
        embedding_year_dim=args.embedding_dim_year,
        hidden_dim=args.hidden_dim,
        sequence_length=args.sequence_length,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        dropout_rate=args.dropout_rate,
        reuse_item_embeddings=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), 2e-4, betas=(0.9, 0.999), weight_decay=5e-5
    )
    model = model.to(args.device)
    
    return model, optimizer, train_loader, num_items


def train_batch(batch, model, optimizer, num_items, tracks, args):
    optimizer.zero_grad()
    inputs = batch["input_seq"].to(args.device)  # (batch, seq_len)
    labels = batch["label_seq"].to(args.device)  # (batch)
    negatives = batch["negatives_seq"].to(args.device)  # (batch, n_negatives)
    inbatch_negatives = batch["in_batch_negatives_seq"].to(args.device)
    inputs_genre = batch["input_genre"].to(
        args.device
    )  # (batch, seq_len, max_genre_per_tracks)
    inputs_domain = batch["input_domain"].to(args.device)  # (batch, seq_len)
    inputs_year = batch["input_year"].to(args.device)  # (batch, seq_len)
    labels_genre = batch["label_genre"].to(args.device)  # (batch, max_genre_per_tracks)
    labels_domain = batch["label_domain"].to(args.device)  # (batch)
    labels_year = batch["label_year"].to(args.device)  # (batch)
    negatives_genre = batch["negatives_genre"].to(
        args.device
    )  # (batch, n_negatives, max_genre_per_tracks)
    negatives_domain = batch["negatives_domain"].to(args.device)  # (batch, n_negatives)
    negatives_year = batch["negatives_year"].to(args.device)  # (batch, n_negatives)
    inbatch_negatives_genre = batch["in_batch_negatives_genre"].to(
        args.device
    )  # (batch, batch-1, max_genre_per_tracks)
    inbatch_negatives_domain = batch["in_batch_negatives_domain"].to(
        args.device
    )  # (batch, batch-1)
    inbatch_negatives_year = batch["in_batch_negatives_year"].to(
        args.device
    )  # (batch, batch-1)
    popularity_logits_cpu = batch["popularity_logits"]

    output_item_embedding, output_genre_embedding, output_domain_embedding, output_year_embedding = model.get_output_embeddings()

    # negatives와 in-batch negatives 결합
    negatives = torch.cat(
        [negatives, inbatch_negatives], dim=1
    )  # (batch, n_negatives + batch)
    negatives_domain = torch.cat(
        [negatives_domain, inbatch_negatives_domain], dim=1
    )  # (batch, n_negatives + batch)
    negatives_genre = torch.cat(
        [negatives_genre, inbatch_negatives_genre], dim=1
    )  # (batch, n_negatives + batch, max_genre_per_tracks)
    negatives_year = torch.cat(
        [negatives_year, inbatch_negatives_year], dim=1
    )  # (batch, n_negatives + batch)

    last_hidden_state, _ = model(inputs, inputs_genre, inputs_domain, inputs_year)
    last_hidden_vector = last_hidden_state[:, -1, :]  # (batch_size, embedding_dim)

    # Positive (label) 임베딩
    labels_embeddings = output_item_embedding(
        labels.unsqueeze(1)
    )  # (batch, 1, embedding_dim)
    labels_genre_embeddings = output_genre_embedding(
        labels_genre.unsqueeze(1)
    )  # (batch, 1 genre_embedding_dim)
    labels_domain_embeddings = output_domain_embedding(
        labels_domain.unsqueeze(1)
    )  # (batch, 1 domain_embedding_dim)
    labels_year_embeddings = output_year_embedding(
        labels_year.unsqueeze(1)
    )  # (batch, 1, year_embedding_dim)

    # Negative 임베딩
    negatives_embeddings = output_item_embedding(
        negatives
    )  # (batch, n_negatives + batch, embedidng_dim )
    negatives_genre_embeddings = output_genre_embedding(
        negatives_genre
    )  # (batch, n_negatives + batch, genre_embedidng_dim )
    negatives_domain_embeddings = output_domain_embedding(
        negatives_domain
    )  # (batch, n_negatives + batch, domain_embedidng_dim )
    negatives_year_embeddings = output_year_embedding(
        negatives_year
    )  # (batch, n_negatives + batch, year_embedidng_dim )

    # 모든 임베딩 결합
    labels_combined = model.feature_fc(torch.cat(
        [
            labels_embeddings,
            labels_genre_embeddings,
            labels_domain_embeddings,
            labels_year_embeddings,
        ],
        dim=-1,
    ))  # (batch, 1, total_embedding_dim)
    negatives_combined = model.feature_fc(torch.cat(
        [
            negatives_embeddings,
            negatives_genre_embeddings,
            negatives_domain_embeddings,
            negatives_year_embeddings,
        ],
        dim=-1,
    ))  # (batch, n_negatives + batch, total_embedding_dim )
    pos_neg_embeddings = torch.cat(
        [labels_combined, negatives_combined], dim=1
    )  # (batch, n_negatives + batch + 1, total_embedding_dim )

    logits = torch.einsum(
        "bse, bsne -> bsn",
        last_hidden_vector.unsqueeze(1),
        pos_neg_embeddings.unsqueeze(1),
    )  # (batch, 1, n_negatives + batch + 1)

    # popularity-awareness
    popularity_labels = torch.gather(popularity_logits_cpu, 1, labels.cpu().unsqueeze(1)).to(args.device)  # (batch, 1)
    popularity_negatives = torch.gather(popularity_logits_cpu, 1, negatives.cpu()).to(args.device)  # (batch, n_negatives + batch)
    popularity_scores = torch.cat([popularity_labels, popularity_negatives], dim=1)  # (batch, n_negatives + batch + 1)
    original_logits = logits.clone().detach()
    logits = logits + popularity_scores.unsqueeze(1)

    # 정답 후보는 첫번째 항목으로 설정
    gt = torch.zeros_like(logits)
    gt[:, :, 0] = 1

    alpha = negatives.shape[1] / (num_items - 1)
    beta = alpha * ((1 - 1 / alpha) * args.gbce_t + 1 / alpha)
    wandb.log({"train/alpha": alpha, "train/beta": beta})

    positive_logits = logits[:, :, 0:1].to(torch.float64)  # (batch, 1, 1)
    negative_logits = logits[:, :, 1:].to(
        torch.float64
    )  # (batch, 1, num_negatives)

    eps = 1e-10
    # -------------------------------
    # BPR Loss 계산 (padding 후보와 label padding 모두 무시)
    diff = (
        positive_logits - negative_logits
    )  # (batch, 1, num_negatives + batch) - broadcast subtraction
    bpr_loss_elements = -torch.log(torch.sigmoid(diff) + eps)
    # negatives가 padding이 아닌지 체크: (batch, num_negatives) -> (batch, 1, num_negatives)
    neg_mask = (
        (negatives != len(tracks) + 1).float().unsqueeze(1)
    )  # (batch, 1, n_negatives)

    label_mask = (
        (labels != len(tracks) + 1).float().view(labels.size(0), 1, 1)
    )  # (batch, 1, 1)
    bpr_mask = neg_mask * label_mask
    bpr_loss = (bpr_loss_elements * bpr_mask).sum() / (bpr_mask.sum() + eps)
    # -------------------------------

    positive_probs = torch.clamp(torch.sigmoid(positive_logits), eps, 1 - eps)
    positive_probs_adjusted = torch.clamp(
        positive_probs.pow(-beta), 1 + eps, torch.finfo(torch.float64).max
    )
    to_log = torch.clamp(
        torch.div(1.0, (positive_probs_adjusted - 1)),
        eps,
        torch.finfo(torch.float64).max,
    )
    positive_logits_transformed = to_log.log()  # (batch, 1, 1)

    if torch.isnan(positive_logits_transformed).any():
        print("WARNING: NaN detected in positive_logits_transformed!")
    if torch.isinf(positive_logits_transformed).any():
        print("WARNING: Inf detected in positive_logits_transformed!")

    logits = torch.cat(
        [positive_logits_transformed, negative_logits], -1
    )  # (batch, 1, num_negatives + batch + 1)

    # -------------------------------
    # BCE Loss 계산 시, padding된 negatives와 label이 padding인 경우는 제외
    negatives_mask = (
        negatives != num_items + 1
    ).float()  # (batch, n_negatives + batch)
    candidate_mask = torch.cat(
        [torch.ones((negatives_mask.shape[0], 1), device=args.device), negatives_mask],
        dim=1,
    )  # (batch, num_negatives + batch + 1)
    candidate_mask = candidate_mask.unsqueeze(
        1
    )  # (batch, 1, num_negatives + batch + 1)
    # label이 padding인 경우 해당 배치 전체에 대해 loss 무시 (broadcast)
    candidate_mask = candidate_mask * (labels != num_items + 1).float().view(
        labels.size(0), 1, 1
    )
    # -------------------------------

    loss_per_element = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, gt, reduction="none"
    )
    loss_per_element = loss_per_element * candidate_mask
    loss = loss_per_element.sum() / (candidate_mask.sum() + eps)
    loss += bpr_loss

    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss, original_logits, labels, negatives


def train(model, optimizer, train_loader, num_items, tracks, args):
    for epoch in range(args.epochs):
        model.train()
        
        for batch_idx, batch in enumerate(train_loader):
            loss, original_logits, labels, negatives = train_batch(batch, model, optimizer, num_items, tracks, args)
            wandb.log({"train/loss": loss.item()})
            
            top1_acc, recall_at_k, ndcg_at_k = compuate_metrics(original_logits, labels, num_items)
            wandb.log({"train/top1_acc": top1_acc, "train/recall_at_k": recall_at_k, "train/ndcg_at_k": ndcg_at_k})

        model_name = f"gsasrec-{epoch}.pt"
        torch.save(model.state_dict(), model_name)
