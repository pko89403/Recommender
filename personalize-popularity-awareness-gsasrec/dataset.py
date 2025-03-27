from datasets import load_dataset
from torch.utils.data import Dataset
import random
import torch
import numpy as np
import wandb


class SequenceDataset(Dataset):
    def __init__(
        self,
        raw_dataset_path,
        track_meta_dict,
        padding_value,
        neg_sampling_prob,
        max_length=200,
        num_negatives=256,
        max_genres_per_track=5,
        eps=0.01,
    ):
        self.inputs = load_dataset(
            "parquet", data_files=raw_dataset_path, streaming=False
        )["train"]
        self.track_meta_dict = track_meta_dict
        self.max_length = max_length
        self.neg_sampling_prob = neg_sampling_prob
        self.padding_value = padding_value
        self.num_negatives = num_negatives
        self.item_list = np.arange(1, self.padding_value)
        self.max_genres_per_track = max_genres_per_track
        self.eps = eps

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inp = self.inputs[idx]
        inp["actions"].sort(key=lambda x: (x["create_dtime"]))

        seq, genre_seq, domain_seq, year_seq = [], [], [], []
        for x in inp["actions"]:
            if x["rating"] > 0:
                seq.append(x["track_index"])
                meta = self.track_meta_dict[x["track_index"]]
                genre_seq.append(meta["genre_id_list"])
                domain_seq.append(meta["domain"])
                year_seq.append(meta["year_category"])

        pad_meta = self.track_meta_dict[self.padding_value]

        # 최소 길이 체크 (최소 2개는 있어야 input과 label로 나눌 수 있음)
        if len(seq) < 2:
            pad_len = self.max_length + 1 - len(seq)
            seq = [self.padding_value] * pad_len + seq
            genre_seq = [pad_meta["genre_id_list"]] * pad_len + genre_seq
            domain_seq = [pad_meta["domain"]] * pad_len + domain_seq
            year_seq = [pad_meta["year_category"]] * pad_len + year_seq
        else:
            # 랜덤한 위치 선택 (최소 1 이상이어야 함)
            borderline = random.randint(1, len(seq) - 1)
            seq = seq[: borderline + 1]  # 랜덤 위치까지 자르기
            genre_seq = genre_seq[: borderline + 1]
            domain_seq = domain_seq[: borderline + 1]
            year_seq = year_seq[: borderline + 1]

            # max_length 맞추기 위한 패딩
            if len(seq) < self.max_length + 1:
                pad_len = self.max_length + 1 - len(seq)
                seq = [self.padding_value] * pad_len + seq
                genre_seq = [pad_meta["genre_id_list"]] * pad_len + genre_seq
                domain_seq = [pad_meta["domain"]] * pad_len + domain_seq
                year_seq = [pad_meta["year_category"]] * pad_len + year_seq

            else:
                seq = seq[-(self.max_length + 1) :]
                genre_seq = genre_seq[-(self.max_length + 1) :]
                domain_seq = domain_seq[-(self.max_length + 1) :]
                year_seq = year_seq[-(self.max_length + 1) :]

        # 네거티브 샘플 생성 (전체 시퀀스 내 아이템 제외)
        user_history_set = set(seq)
        user_history_set.discard(self.padding_value)

        negatives = set()
        while len(negatives) < self.num_negatives:
            n_needed = self.num_negatives - len(negatives)
            # 아이템 인덱스 범위: 1 ~ NUM_ITEMS (padding 값은 NUM_ITEMS+1)
            candidates = np.random.choice(
                self.item_list, size=n_needed * 2, replace=False, p=self.neg_sampling_prob
            )
            for c in candidates:
                if c not in user_history_set and c not in negatives:
                    negatives.add(int(c))
                if len(negatives) >= self.num_negatives:
                    break
                
        negatives_genre = []
        negatives_domain = []
        negatives_year = []
        for negative in negatives:
            meta = self.track_meta_dict[negative]
            negatives_genre.append(meta["genre_id_list"])
            negatives_domain.append(meta["domain"])
            negatives_year.append(meta["year_category"])
            
        input_seq = torch.tensor(seq[:-1], dtype=torch.long) # (seq_len)
        # ===============================
        # popularity_logits 계산 로직 (get_last_time_popularity_logits 내장)
        seq_tensor = input_seq.unsqueeze(0)  # (1, seq_len)
        
        # 아이템 등장 횟수 계산
        seq_tensor = torch.where(
            (seq_tensor >= 0) & (seq_tensor < self.padding_value),
            seq_tensor,
            torch.full_like(seq_tensor, self.padding_value)
        )
        counts = torch.zeros((1, self.padding_value + 1), dtype=torch.float32, requires_grad=False)
        counts.scatter_add_(1, seq_tensor, torch.ones_like(seq_tensor, dtype=torch.float32))
        counts = counts + self.eps
        counts[:, self.padding_value:] = self.eps  # padding 및 special token 위치는 eps로

        # 정규화: 우선 최대값으로 나눈 후 valid 아이템들의 합으로 다시 정규화
        probs = counts / counts.max(dim=-1, keepdim=True).values
        norm_probs = probs / probs[:, :self.padding_value].sum(dim=-1, keepdim=True)

        # logit 변환: log(p/(1-p))
        popularity_logits = -(torch.log1p(-norm_probs) - torch.log(norm_probs))
        popularity_logits = popularity_logits.squeeze(0) # (self.padding_value+1,)
        
        return {
            "input_seq": input_seq,  # 입력은 마지막 아이템 제외
            "input_genre": torch.tensor(genre_seq[:-1], dtype=torch.long),  # 입력은 마지막 아이템 제외
            "input_domain": torch.tensor(domain_seq[:-1], dtype=torch.long),  # 입력은 마지막 아이템 제외
            "input_year": torch.tensor(year_seq[:-1], dtype=torch.long),  # 입력은 마지막 아이템 제외
            "label_seq": torch.tensor(seq[-1], dtype=torch.long),  # 정답은 마지막 아이템 하나만
            "label_genre": torch.tensor(genre_seq[-1], dtype=torch.long),  # 입력은 마지막 아이템 제외
            "label_domain": torch.tensor(domain_seq[-1], dtype=torch.long),  # 입력은 마지막 아이템 제외
            "label_year": torch.tensor(year_seq[-1], dtype=torch.long),  # 입력은 마지막 아이템 제외
            "negatives_seq": torch.tensor(list(negatives), dtype=torch.long),
            "negatives_genre": torch.tensor(negatives_genre, dtype=torch.long),
            "negatives_domain": torch.tensor(negatives_domain, dtype=torch.long),
            "negatives_year": torch.tensor(negatives_year, dtype=torch.long),
            "popularity_logits": popularity_logits
        }


def collate_with_inbatch_negatives(input_batch, padding_value, track_meta_dict):
    # Extract sequence inputs, labels, and negatives
    batch_inputs = torch.stack([item["input_seq"] for item in input_batch], dim=0)
    batch_labels = torch.stack([item["label_seq"] for item in input_batch], dim=0)
    negatives = torch.stack([item["negatives_seq"] for item in input_batch], dim=0)
    popularity_logits = torch.stack([item["popularity_logits"] for item in input_batch], dim=0)

    # Extract genre, domain, and year inputs, labels, and negatives
    batch_genre_inputs = torch.stack(
        [item["input_genre"] for item in input_batch], dim=0
    )
    batch_domain_inputs = torch.stack(
        [item["input_domain"] for item in input_batch], dim=0
    )
    batch_year_inputs = torch.stack([item["input_year"] for item in input_batch], dim=0)

    batch_genre_labels = torch.stack(
        [item["label_genre"] for item in input_batch], dim=0
    )
    batch_domain_labels = torch.stack(
        [item["label_domain"] for item in input_batch], dim=0
    )
    batch_year_labels = torch.stack([item["label_year"] for item in input_batch], dim=0)

    negatives_genre = torch.stack(
        [item["negatives_genre"] for item in input_batch], dim=0
    )
    negatives_domain = torch.stack(
        [item["negatives_domain"] for item in input_batch], dim=0
    )
    negatives_year = torch.stack(
        [item["negatives_year"] for item in input_batch], dim=0
    )

    batch_size = batch_labels.size(0)

    # 다른 샘플들의 input만을 사용하여 후보 풀 구성 (각 모달리티별)
    # candidates_seq를 batch_labels 기반으로 구성 (shape: (B, B))
    candidates_seq = batch_labels.unsqueeze(0).expand(batch_size, batch_size)
    candidates_genre = batch_genre_labels.unsqueeze(0).expand(
        batch_size, batch_size, -1
    )  # (B, B, max_genre_per_tracks)
    candidates_domain = batch_domain_labels.unsqueeze(0).expand(
        batch_size, batch_size
    )  # (B, B)
    candidates_year = batch_year_labels.unsqueeze(0).expand(
        batch_size, batch_size
    )  # (B, B)

    # ----- 마스킹: 각 샘플의 후보가 해당 샘플의 입력 시퀀스 내에 존재하거나, 레이블과 동일하면 마스킹 -----
    # 1) 시퀀스의 경우: candidates_seq (B, B)와 batch_inputs (B, seq_len)
    #    각 샘플 i에서, candidates_seq[i, j]가 batch_inputs[i, :] 중 어느 토큰과 같으면 True.
    mask_input = (candidates_seq.unsqueeze(2) == batch_inputs.unsqueeze(1)).any(
        dim=2
    )  # (B, B)
    mask_label = candidates_seq == batch_labels.unsqueeze(1)
    mask = mask_input | mask_label  # shape: (batch_size, batch_size)

    # mask된 후보는 모든 모달리티에서 padding_value로 대체
    candidates_seq = candidates_seq.clone()
    candidates_genre = candidates_genre.clone()
    candidates_domain = candidates_domain.clone()
    candidates_year = candidates_year.clone()

    pad_meta = track_meta_dict[padding_value]

    candidates_seq[mask] = torch.tensor(padding_value, dtype=torch.long)
    candidates_genre[mask] = torch.tensor(pad_meta["genre_id_list"], dtype=torch.long)
    candidates_domain[mask] = torch.tensor(pad_meta["domain"], dtype=torch.long)
    candidates_year[mask] = torch.tensor(pad_meta["year_category"], dtype=torch.long)

    # 자기 자신의 후보(대각원소)는 제거: off-diagonal mask 사용
    off_diag_mask = ~torch.eye(batch_size, dtype=torch.bool, device=batch_inputs.device)
    in_batch_negatives_seq = candidates_seq[off_diag_mask].view(
        batch_size, batch_size - 1
    )
    in_batch_negatives_genre = candidates_genre[off_diag_mask].view(
        batch_size, batch_size - 1, -1
    )
    in_batch_negatives_domain = candidates_domain[off_diag_mask].view(
        batch_size, batch_size - 1
    )
    in_batch_negatives_year = candidates_year[off_diag_mask].view(
        batch_size, batch_size - 1
    )

    return {
        "input_seq": batch_inputs,
        "label_seq": batch_labels,
        "negatives_seq": negatives,
        "input_genre": batch_genre_inputs,
        "label_genre": batch_genre_labels,
        "negatives_genre": negatives_genre,
        "input_domain": batch_domain_inputs,
        "label_domain": batch_domain_labels,
        "negatives_domain": negatives_domain,
        "input_year": batch_year_inputs,
        "label_year": batch_year_labels,
        "negatives_year": negatives_year,
        "in_batch_negatives_seq": in_batch_negatives_seq,
        "in_batch_negatives_genre": in_batch_negatives_genre,
        "in_batch_negatives_domain": in_batch_negatives_domain,
        "in_batch_negatives_year": in_batch_negatives_year,
        "popularity_logits": popularity_logits,
    }