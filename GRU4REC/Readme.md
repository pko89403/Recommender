# 새로운 세션이 나올 때 마다 가중치를 초기화 한ㄴ다.
> 결국에는 hidden state를 Recursive하게 넣어주게 된다.
# Mini-Batch에서 세션을 채워 넣는다.
# Mini-Batch 단에서 Negative Sampling을 한다.
> collate_fn을 사용하는 건 어떨까?