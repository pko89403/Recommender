
# Paragraph2Vec(DM) 모델 기반 추천 모델
word2vec 이 아닌 paragraph2vec(doc2vec) 기반으로 paragraph(item_meta_info)를 포함해서 item latent factor를 학습한다.

### Data
title_id(1) - title(N) - titleIdx(N)
아...이거 title_id( 제품 코드 ) -> titleIdx( 임베딩 뽑아내려고 쓰는 제품 인덱스 )

### paragraph2vec(DM) 모델 Forward Shape
~~~python
train_data(product=tensor([0, 0]), target=tensor([0, 1]), context=[tensor([1, 2]), tensor([2, 3])])
torch.Size([1, 2, 3])
item_id_embedding :  tensor([[[0.5681, 0.9501, 1.1617],
         [0.5681, 0.9501, 1.1617]]], grad_fn=<UnsqueezeBackward0>)
torch.Size([2, 2])
context_ids :  tensor([[1, 2],
        [2, 3]])
torch.Size([2, 2, 3])
context_ids_embedding :  tensor([[[-0.8951, -0.2582,  0.0040],
         [-1.6003, -0.2306, -1.2729]],

        [[-1.6003, -0.2306, -1.2729],
         [ 1.4311, -0.3718, -0.6798]]], grad_fn=<EmbeddingBackward>)
torch.Size([3, 2, 3])
total_embedding :  tensor([[[ 0.5681,  0.9501,  1.1617],
         [ 0.5681,  0.9501,  1.1617]],

        [[-0.8951, -0.2582,  0.0040],
         [-1.6003, -0.2306, -1.2729]],

        [[-1.6003, -0.2306, -1.2729],
         [ 1.4311, -0.3718, -0.6798]]], grad_fn=<CatBackward>)
torch.Size([1, 2, 3])
tensor([[[-0.6424,  0.1538, -0.0357],
         [ 0.1330,  0.1159, -0.2637]]], grad_fn=<MeanBackward1>)
~~~