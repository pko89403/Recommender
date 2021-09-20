# BERT4REC 
https://arxiv.org/abs/1904.06690
## Dataset
MovieLens 

## Requirements
- pytorch-lightning
- pytoch
- pandas 
- numpy

## RUN
```sh
python main.py
```

## Project Architecture 
```sh
.
├── Readme.md
├── artifacts
│   └── lightning_logs
│       └── version_3
│           ├── checkpoints
│           │   └── epoch=49-step=149.ckpt
│           ├── events.out.tfevents.1632112817.Seokwooui-MacBookPro.local.85309.0
│           ├── events.out.tfevents.1632114261.Seokwooui-MacBookPro.local.85309.1
│           └── hparams.yaml
├── datamodules.py
├── dataset.py
├── loss.py
├── metrics.py
├── models.py
├── preprocessing.ipynb
├── main.py
└── utils.py
```

# Network Architecture
```sh
BERT4REC(
  (item_embeddings): Embedding(10000, 32)
  (pos_embeddings): Embedding(512, 32)
  (transformer_encoder): TransformerEncoder(
    (layers): ModuleList(
      (0): TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=32, out_features=32, bias=True)
        )
        (linear1): Linear(in_features=32, out_features=2048, bias=True)
        (dropout): Dropout(p=0.8, inplace=False)
        (linear2): Linear(in_features=2048, out_features=32, bias=True)
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.8, inplace=False)
        (dropout2): Dropout(p=0.8, inplace=False)
      )
      ...
      (n): TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=32, out_features=32, bias=True)
        )
        (linear1): Linear(in_features=32, out_features=2048, bias=True)
        (dropout): Dropout(p=0.8, inplace=False)
        (linear2): Linear(in_features=2048, out_features=32, bias=True)
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.8, inplace=False)
        (dropout2): Dropout(p=0.8, inplace=False)
      )
    )
  )
  (out): Linear(in_features=32, out_features=10000, bias=True)
)
```