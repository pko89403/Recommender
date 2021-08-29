# ElasticSearch에서 벡터 연산을 하면 빠르디
```
    es-dotProduct-Example.ipynb # es에서 dotProduct를 하는 예제 
    Factorization.ipynb # 토치 모델(FM)로 임베딩 생성 후 es document 생성 후 vector search
    run_elastic.sh # docker-compose.yml로 single-node로 구성한 es와 kibana
```
엘라스틱 서치 벡터 매핑 인덱스 생성
```json
{
  "mappings": {
    "properties": {
      "feature_type":{
        "type":"keyword"
      },
      "embedding": {
        "type": "dense_vector",
        "dims": 120 # 벡터 차원 수 
      },
      "bias": {
        "type":"double"
      },
      "title" : {
        "type" : "keyword"
      }
    }
  }
}
```
좀 억지로 vector 연산 결과가 마이너스를 안나오게 만들었다. (나오면 400 에러가 나와서 안되기 때문에 완전 억지로 만들었음)
```json
{
    "query": {
    "script_score": {
      "query" : {
          "bool" : {
          "filter" : {
            "term" : {
              "feature_type" : "movie" 
            }
          }
        }
      },
      "script": {
        "source": "cosineSimilarity(params.query_vector, 'embedding') + doc['bias'].value", 
        # "source": "dotProduct(params.query_vector, 'embedding') + doc['bias'].value", 
        # "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0", 
        "params": {
          "query_vector": v_metadata
        }
      }
    }
  }
}
```

## 참고 사이트
- https://haandol.github.io/2020/02/28/elasticsearch-dense-vector-consinesimilarity.html
- https://yonigottesman.github.io/recsys/pytorch/elasticsearch/2020/02/18/fm-torch-to-recsys.html