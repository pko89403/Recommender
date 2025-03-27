#!/usr/bin/env python
# coding: utf-8

import argparse
from pps_gsasrec import initialize_training, train


def parse_args():
    parser = argparse.ArgumentParser(description="Train GSASRec model with given parameters")
    # 파일 경로 인자
    parser.add_argument("--dataset_name", type=str, default="interactions.parquet",
                        help="Path to the training dataset parquet files")
    parser.add_argument("--track_name", type=str, default="items.parquet",
                        help="Path to the tracks metadata parquet files")
    parser.add_argument("--character_name", type=str, default="characters.parquet",
                        help="Path to the characters parquet files")
    # 학습/모델 하이퍼파라미터
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--embedding_dim_id", type=int, default=122, help="Dimension of ID embedding")
    parser.add_argument("--embedding_dim_genre", type=int, default=22, help="Dimension of genre embedding")
    parser.add_argument("--embedding_dim_domain", type=int, default=6, help="Dimension of domain embedding")
    parser.add_argument("--embedding_dim_year", type=int, default=12, help="Dimension of year embedding")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--num_heads", type=int, default=1, help="Number of heads")
    parser.add_argument("--num_blocks", type=int, default=4, help="Number of transformer blocks")
    parser.add_argument("--gbce_t", type=float, default=0.75, help="GBCE temperature")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--sequence_length", type=int, default=100, help="Sequence length")
    parser.add_argument("--n_negatives", type=int, default=128, help="Number of negatives")
    parser.add_argument("--max_genres_per_track", type=int, default=5, help="Max genres per track")
    parser.add_argument("--ips_temperature", type=float, default=0.75, help="IPS temperature")
    parser.add_argument("--pps_eps", type=float, default=0.01, help="PPS epsilon")
    parser.add_argument("--num_workers", type=int, default=15, help="Number of workers for DataLoader")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    return parser.parse_args()


def main():
    args = parse_args()
    model, optimizer, train_loader, num_items, tracks = initialize_training(args)
    train(model, optimizer, train_loader, num_items, tracks, args)


if __name__ == "__main__":
    main()
