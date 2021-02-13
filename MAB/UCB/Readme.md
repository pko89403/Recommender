# Optimized action based on confidence bounds
간소화된 강화학습 태스크. 멀티 암드 밴딧은 k 개의 슬롯 머신 들이 있고 모두 각각의 보상 확률을 가지고 있을 때, 최고인 하나를 찾을 때 사용한다.

밴딧은 Exploration 과 Exploitation 간의 트레이드-오프 간의 밸런스를 고려해야 한다. 우선 보상에 대한 사전 지식이 없기 때문이다. 많은 암에 대해 시도를 해야하고(Exploration), 좋은 암을 반복적으로 당겨야한다(Exploitation. 가장 간단한 방법은 그리디 방법으로 입실론 그리디 방법이 있다.

입실론 그리디 방법 보다 더 좋은 방법이 있는데 Upper Confidence Bound(UCB)이다.


```
    python agent.py
```


- [참고 사이트 1](https://towardsdatascience.com/multi-armed-bandits-ucb-algorithm-fa7861417d8c)