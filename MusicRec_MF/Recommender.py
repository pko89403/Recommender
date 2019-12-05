# https://github.com/wolfecameron/music_recommendation/blob/master/music_ratings_learner.ipynb
import torch
import torch.nn

class Recommender(nn.Module):
    def __init__(self, num_users, num_artists, num_factors):
        super().__init__()
        # 두 개의 임베딩 매트릭스를 선언한다.
        # nn.Embedding(총 유저 수, 임베딩 시킬 벡터의 차원) 총 유저에 대해 해당 차원 만큼 임베딩 매트릭스를 생성함.
        self.u = nn.Embedding(num_users, num_factors)
        self.a = nn.Embedding(num_artists, num_factors)
        self.u.weight.data.uniform_(-.01, .01)
        self.a.weight.data.uniform_(-.01, .01)
        # bias 를 추가한다.
        self.ub = nn.Embedding(num_users, 1)
        self.ab = nn.Embedding(num_artists, 1)
        self.ub.weight.data.uniform_(-.01, .01)
        self.ab.weight.data.uniform_(-.01, .01)

    def forward(self, cats, conts):
        # 해당 인덱스에 위치하는 유저, 아티스트 두 벡터를 가져온다.
        # 내적해서 rating을 예측한다.
        users, artists = cats[:, 0], cats[:, 1]
        us, art = self.u(users), self.a(artists)
        dp = (us * art).sum(1)
        # Bias 추가한 것을 더한다. 혓
        dpb = dp + self.ub(users).squeeze() + self.ab(artists).squeeze()
        return dpb