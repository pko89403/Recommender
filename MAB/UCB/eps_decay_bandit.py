import numpy as np 
 

class model:
    def __init__(self, k, param, iters, mu='random'):
        self.k = k # 사용할 암(슬롯) 수
        self.param = param # Exploration할 확률 
        self.iters = iters # 전체 iteration 수
        self.n = 0 # Step Count 수
        self.k_n = np.zeros(k) # 암 각각의 Step Count 수
        self.mean_reward = 0 # 전체 평균 보상
        self.reward = np.zeros(iters) # Iteration 마다의 평균 보상
        self.k_mean_reward = np.zeros(k) # 암 각각의 보상 평균

        # 각 암의 보상에 대한 확률을 초기화 한다.
        if type(mu) == list or type(mu).__module__ == np.__name__:
            # 유저 정의 평균
            self.mu = np.array(mu)
        elif mu == 'random':
            # 확률 분포를 통해 생성
            self.mu = np.random.normal(0, 1, k)
        elif mu == 'sequence':
            # 각 암마다 평균을 하나 씩 증가 시킨다.
            self.mu = np.linspace(0,k-1,k)
    
    def __str__(self):
        return 'eps-decay'

    def pull(self):
        prob = np.random.rand() # 랜덤 넘버를 생성한다.

        # 암을 선택한다.
        if prob < 1 / ( 1 + self.n / self.k):
            # 암을 랜덤하게 선택한다.
            a = np.random.choice(self.k)
        else:
            # 암을 그리디하게 선택한다.
            a = np.argmax(self.k_mean_reward)

        # 선택된 암의 보상
        reward = np.random.normal(self.mu[a], 1)

        # 상태 업데이트
        self.n += 1
        self.k_n[a] += 1
        self.mean_reward = self.mean_reward + ( reward - self.mean_reward ) / self.n 
        self.k_mean_reward[a] = self.k_mean_reward[a] + ( reward - self.k_mean_reward[a] ) / self.k_n[a]

    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = self.mean_reward

    def reset(self, mu):
        self.n = 0
        self.k_n = np.ones(self.k)
        self.mean_reward = 0
        self.reward = np.zeros(self.iters)
        self.k_mean_reward = np.zeros(self.k)

        # 각 암의 보상에 대한 확률을 초기화 한다.
        if type(mu) == list or type(mu).__module__ == np.__name__:
            # 유저 정의 평균
            self.mu = np.array(mu)
        elif mu == 'random':
            # 확률 분포를 통해 생성
            self.mu = np.random.normal(0, 1, self.k)
        elif mu == 'sequence':
            # 각 암마다 평균을 하나 씩 증가 시킨다.
            self.mu = np.linspace(0, self.k-1, self.k)        