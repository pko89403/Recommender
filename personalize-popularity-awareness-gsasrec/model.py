import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout_rate=0.5):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.val_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)  # Change the dropout rate as needed

    def forward(self, queries, keys, causality=False):
        Q = self.query_proj(queries)
        K = self.key_proj(keys)
        V = self.val_proj(keys)

        # Split and concat
        Q_ = torch.cat(Q.chunk(self.num_heads, dim=2), dim=0)
        K_ = torch.cat(K.chunk(self.num_heads, dim=2), dim=0)
        V_ = torch.cat(V.chunk(self.num_heads, dim=2), dim=0)

        # Multiplication
        outputs = torch.matmul(Q_, K_.transpose(1, 2))

        # Scale
        outputs = outputs / (K_.size(-1) ** 0.5)

        # Key Masking
        key_masks = torch.sign(torch.sum(torch.abs(keys), dim=-1))
        key_masks = key_masks.repeat(self.num_heads, 1)
        key_masks = key_masks.unsqueeze(1).repeat(1, queries.size(1), 1)

        outputs = outputs.masked_fill(key_masks == 0, float("-inf"))

        # Causality
        if causality:
            diag_vals = torch.ones_like(outputs[0])
            tril = torch.tril(diag_vals)
            masks = tril[None, :, :].repeat(outputs.size(0), 1, 1)

            outputs = outputs.masked_fill(masks == 0, float("-inf"))

        # Activation
        outputs = F.softmax(outputs, dim=-1)
        outputs = torch.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)

        # Query Masking
        query_masks = torch.sign(torch.sum(torch.abs(queries), dim=-1))
        query_masks = query_masks.repeat(self.num_heads, 1)
        query_masks = query_masks.unsqueeze(-1).repeat(1, 1, keys.size(1))

        outputs *= query_masks
        attention_chunks = outputs.chunk(self.num_heads, dim=0)
        attention_weights = torch.stack(attention_chunks, dim=1)

        # Dropouts
        outputs = self.dropout(outputs)

        # Weighted sum
        outputs = torch.matmul(outputs, V_)

        # Restore shape
        outputs = torch.cat(outputs.chunk(self.num_heads, dim=0), dim=2)
        return outputs, attention_weights


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, dropout_rate=0.5, causality=True):
        super(TransformerBlock, self).__init__()

        self.first_norm = nn.LayerNorm(dim)
        self.second_norm = nn.LayerNorm(dim)

        self.multihead_attention = MultiHeadAttention(dim, num_heads, dropout_rate)

        self.dense1 = nn.Linear(dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.causality = causality

    def forward(self, seq, mask=None):
        x = self.first_norm(seq)
        queries = x
        keys = seq
        x, attentions = self.multihead_attention(queries, keys, self.causality)

        # Add & Norm
        x = x + queries
        x = self.second_norm(x)

        # Feed Forward
        residual = x
        x = self.dense1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)

        # Add & Norm
        x = x + residual

        # Apply mask if provided
        if mask is not None:
            x *= mask

        return x, attentions


class MultiMetaAggregator(torch.nn.Module):
    """
    여러 장르를 가진 트랙의 장르 임베딩을 결합하는 모듈입니다.
    aggregation 인자에 따라 'mean', 'sum', 'attention' 방식 중 하나로 결합합니다.
    """

    def __init__(self, num_category, embedding_dim, aggregation="mean", padding_idx=0):
        super(MultiMetaAggregator, self).__init__()
        self.aggregation = aggregation
        self.meta_embedding = torch.nn.Embedding(
            num_category, embedding_dim, padding_idx=padding_idx
        )

        torch.nn.init.trunc_normal_(self.meta_embedding.weight)

        if self.aggregation == "attention":
            # Attention weights projection layer
            self.attention_layer = torch.nn.Linear(embedding_dim, 1)

    def forward(self, meta_indices):
        """
        Args:
          meta_indices (LongTensor): (batch, sequence_length, num_meta_per_track) 각 트랙의 메타 인덱스 (패딩: padding_idx)
        Returns:
          aggregated (Tensor): (batch, sequence_length, embedding_dim) 결합된 임베딩 벡터
        """

        # Step 1: Embed meta indices
        meta_embeddings = self.meta_embedding(
            meta_indices
        )  # (batch, sequence_length, num_meta_per_track, embedding_dim)

        # Step 2: Apply aggregation method
        if self.aggregation == "mean":
            # Compute mean along the num_meta_per_track dimension
            aggregated = meta_embeddings.mean(
                dim=-2
            )  # (batch, sequence_length, embedding_dim)

        elif self.aggregation == "sum":
            # Compute sum along the num_meta_per_track dimension
            aggregated = meta_embeddings.sum(
                dim=-2
            )  # (batch, sequence_length, embedding_dim)

        elif self.aggregation == "attention":
            # Compute attention weights
            attention_weights = torch.softmax(
                self.attention_layer(meta_embeddings), dim=-2
            )  # (batch, sequence_length, num_meta_per_track, 1)

            # Apply attention weights to embeddings
            aggregated = torch.sum(
                meta_embeddings * attention_weights, dim=-2
            )  # (batch, sequence_length, embedding_dim)

        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation}")

        return aggregated  # (batch, sequence_length, embedding_dim)


class GSASRec(torch.nn.Module):
    def __init__(
        self,
        num_items,
        padding_value,
        embedding_id_dim,
        genre_mapping,
        embedding_genre_dim,
        domain_mapping,
        embedding_domain_dim,
        year_category_mapping,
        embedding_year_dim,
        hidden_dim,
        sequence_length=200,
        num_heads=4,
        num_blocks=3,
        dropout_rate=0.5,
        reuse_item_embeddings=False,
    ):
        super(GSASRec, self).__init__()
        self.num_items = num_items
        self.padding_value = padding_value
        self.embedding_id_dim = embedding_id_dim
        self.embedding_genre_dim = embedding_genre_dim
        self.embedding_domain_dim = embedding_domain_dim
        self.embedding_year_dim = embedding_year_dim
        self.embedding_total_dim = self.embedding_id_dim + self.embedding_genre_dim + self.embedding_domain_dim + self.embedding_year_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads

        self.item_embedding = torch.nn.Embedding(
            self.num_items + 2, self.embedding_id_dim, padding_idx=self.padding_value
        )  # items are enumerated from 1;  +1 for padding
        self.genre_embedding = MultiMetaAggregator(
            len(genre_mapping) + 1, self.embedding_genre_dim, aggregation="mean", padding_idx=0
        )
        self.domain_embedding = torch.nn.Embedding(
            len(domain_mapping) + 1, self.embedding_domain_dim, padding_idx=0
        )
        self.year_embedding = torch.nn.Embedding(
            len(year_category_mapping) + 1, self.embedding_year_dim, padding_idx=0
        )
        self.feature_fc = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_total_dim, self.hidden_dim),
            torch.nn.GELU()
        )
        self.position_embedding = torch.nn.Embedding(
            self.sequence_length + 1, self.hidden_dim
        )
        self.embeddings_dropout = torch.nn.Dropout(self.dropout_rate)

        self.transformer_blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    self.hidden_dim,
                    self.num_heads,
                    self.hidden_dim,
                    dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )
        self.seq_norm = torch.nn.LayerNorm(self.hidden_dim)
        self.reuse_item_embeddings = reuse_item_embeddings
        if not self.reuse_item_embeddings:
            self.output_item_embedding = torch.nn.Embedding(
                self.num_items + 2, self.embedding_id_dim, padding_idx=self.padding_value
            )  # items are enumerated from 1;  +1 for padding
            self.output_genre_embedding = MultiMetaAggregator(
                len(genre_mapping) + 1, self.embedding_genre_dim, aggregation="mean", padding_idx=0
            )
            self.output_domain_embedding = torch.nn.Embedding(
                len(domain_mapping) + 1, self.embedding_domain_dim, padding_idx=0
            )
            self.output_year_embedding = torch.nn.Embedding(
                len(year_category_mapping) + 1, self.embedding_year_dim, padding_idx=0
            )

            torch.nn.init.trunc_normal_(self.output_item_embedding.weight)
            torch.nn.init.trunc_normal_(self.output_domain_embedding.weight)
            torch.nn.init.trunc_normal_(self.output_year_embedding.weight)

        torch.nn.init.trunc_normal_(self.item_embedding.weight)
        torch.nn.init.trunc_normal_(self.domain_embedding.weight)
        torch.nn.init.trunc_normal_(self.year_embedding.weight)
        torch.nn.init.trunc_normal_(self.position_embedding.weight)

    def get_output_embeddings(self):
        if self.reuse_item_embeddings:
            return (
                self.item_embedding,
                self.genre_embedding,
                self.domain_embedding,
                self.year_embedding,
            )
        else:
            return (
                self.output_item_embedding,
                self.output_genre_embedding,
                self.output_domain_embedding,
                self.output_year_embedding,
            )

    # returns last hidden state and the attention weights
    def forward(self, input_id, input_genre, input_domain, input_year):
        input_seq = self.item_embedding(input_id.long())
        genre_seq = self.genre_embedding(input_genre.long())
        domain_seq = self.domain_embedding(input_domain.long())
        year_seq = self.year_embedding(input_year.long())

        seq = torch.cat([input_seq, genre_seq, domain_seq, year_seq], dim=-1)
        seq = self.feature_fc(seq)

        mask = (input_id != self.num_items + 1).float().unsqueeze(-1)

        bs = seq.size(0)
        positions = (
            torch.arange(seq.shape[1]).unsqueeze(0).repeat(bs, 1).to(input_id.device)
        )
        pos_embeddings = self.position_embedding(positions)[: input_id.size(0)]
        seq = seq + pos_embeddings
        seq = self.embeddings_dropout(seq)
        seq *= mask

        attentions = []
        for block in self.transformer_blocks:
            seq, attention = block(seq, mask)
            attentions.append(attention)

        seq_emb = self.seq_norm(seq)
        return seq_emb, attentions

    def extract_item_embeddings(self, track_meta_dict, device=None):
        """
        GSASRec 모델 내장 함수로, 학습된 출력 임베딩을 바탕으로 각 아이템의 최종 임베딩 벡터를 추출합니다.
        
        Args:
            track_meta_dict (dict): 각 트랙의 메타 정보를 담은 딕셔너리.  
                예: { track_index: {"year_category": ..., "domain": ..., "genre_id_list": [...]}, ... }
            max_genres_per_track (int): 각 트랙당 최대 장르 수 (패딩 포함).
            device (str or torch.device, optional): 사용할 디바이스. 미지정 시, 모델 파라미터의 디바이스를 사용.
        
        Returns:
            combined (torch.Tensor): (num_items, embedding_total) 크기의 아이템 임베딩 행렬.
        """
        with torch.inference_mode():
            (output_item_embedding,
             output_genre_embedding,
             output_domain_embedding,
             output_year_embedding) = self.get_output_embeddings()

            # 아이템 임베딩 (인덱스 1부터 num_items까지)
            indices = torch.arange(1, self.num_items + 1, device=device)
            e_item = output_item_embedding(indices) # (num_items, embedding_id)

            # 각 아이템의 메타 정보 추출 (track_meta_dict는 1부터 num_items까지의 키를 가짐)
            genre_list = []
            domain_list = []
            year_list = []
            for i in range(1, self.num_items + 1):
                meta = track_meta_dict[i]
                genre_list.append(meta["genre_id_list"])
                domain_list.append(meta["domain"])
                year_list.append(meta["year_category"])

            # 리스트를 텐서로 변환 (장르는 (num_items, max_genres_per_track))
            genre_indices = torch.tensor(genre_list, dtype=torch.long, device=device)
            domain_indices = torch.tensor(domain_list, dtype=torch.long, device=device)
            year_indices = torch.tensor(year_list, dtype=torch.long, device=device)

            e_genre = output_genre_embedding(genre_indices) # (num_items, embedding_genre)
            e_domain = output_domain_embedding(domain_indices) # (num_items, embedding_domain)
            e_year = output_year_embedding(year_indices) # (num_items, embedding_year)

            combined = torch.cat([e_item, e_genre, e_domain, e_year], dim=-1) # (num_items, embedding_total)
            combined = self.feature_fc(combined)
            return combined