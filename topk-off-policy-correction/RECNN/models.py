import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


"""Actor-Critic 구조
    강화 학습의 MDP 중 Policy와 가치 함수를 딥뉴럴넷으로 파라미터화해서 근사화한 기법
    에이전트의 행동 확률을 직접적으로 학습하는 방법이 불안정하기 때문에, 가치 함수를 같이 써서 안정성을 높인다

    Actor(=Policy) : Policy의 출력이 Action
    Action : 선택하는 Space의 종류에 따라 네가지로 나눠진다
    1. Discrete Stochastic Action : Agent가 취할 수 있는 Action Space가 유한하고 고르는 방법이 확률적임
    2. Discrete Deterministic Action : Agent가 취할 수 있는 Action Space가 유한하고 고르는 방법이 정해져 있음
    3. Continuous Stochastic Action : Agent가 취할 수 있는 Action Space가 무한하고 고르는 방법이 확률적임
    4. Continuous Deterministic Action : Agent가 취할 수 있는 Action Space가 무한하고 고르는 방법이 정해져 있음
    
    Actor : state가 주어질 때 action을 결정한다
    Critic : state의 가치를 평가한다
    
    매 step마다 얻어진 상태(s), 행동(a), 보상(r), 다음 상태(s')를 이용해 모델을 학습한다
"""

class Beta(nn.Module):
    """Learns to resemble historical policy by calculating cross-entropy with action
    
    Args:
        nn (_type_): _description_
    """
    def __init__(self, num_inputs, num_actions):
        super(Beta, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs, num_actions),
            nn.Softmax(dim=1)
        )
        self.optim = torch.optim.AdamW(
            self.net.parameters(),
            lr=1e-5,
            weight_decay=1e-5
        )
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, state, action):
        predicted_action = self.net(state)
        
        loss = self.criterion(predicted_action, action.argmax(1))
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        return predicted_action.detach()

        
        

class DiscreteActor(nn.Module):
    
    def __init__(self, hidden_size, num_inputs, num_actions):
        super(DiscreteActor, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        
        self.saved_log_probs = []
        self.rewards = []
        self.correction = []
        self.lambda_k = []
        
        self.action_source = {"phi": "phi", "beta": "beta"}
        self.select_action = self._select_action
        
        
    def forward(self, inputs):
        """ Policy

        Args:
            inputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = inputs
        x = F.relu(self.linear1(x))
        action_scores = self.linear2(x)
        return F.softmax(action_scores, dim=1)

    def gc(self):
        del self.rewards[:]
        del self.saved_log_probs[:]
        del self.correction[:]
        del self.lambda_k[:]

    def _select_action(self, state, **kwargs):
        """ Policy로 Action을 선택하는 코드

        Args:
            state (_type_): _description_

        Returns:
            _type_: _description_
        """
        # phi
        phi_probs = self.forward(state)
        # policy가 출력한 확률들을 Categorical 메소드로 확률밀도함수로 만듬
        phi_categorical = Categorical(phi_probs)
        # sampling
        phi_action = phi_categorical.sample()
        # action의 로그 확률을 저장
        self.saved_log_probs.append(phi_categorical.log_prob(phi_action))
        return phi_action, phi_probs

    def phi_beta_sample(self, state, beta, action, **kwargs):
        # 1. obtain probabilities
        # note: detach is to block gradient
        beta_probs = beta(state.detach(), action=action)
        phi_probs = self.forward(state)
        
        # 2. probabilities -> categorical distribution
        beta_categorical = Categorical(beta_probs)
        phi_categorical = Categorical(phi_probs)

        # 3. sample the actions
        available_actions = {
            "phi": phi_categorical.sample(),
            "beta": beta_categorical.sample(),
        }
        phi_action = available_actions[self.action_source["phi"]]
        beta_action = available_actions[self.action_source["beta"]]
        
        # 4. calcuate stuff we need
        phi_log_prob = phi_categorical.log_prob(phi_action)
        beta_log_prob = beta_categorical.log_prob(beta_action)
        
        return phi_action, beta_action, phi_log_prob, beta_log_prob, phi_probs

    def _select_action_with_correction(
        self, state, beta, action, **kwargs,
    ):
        phi_action, beta_action, phi_log_prob, beta_log_prob, phi_probs = self.phi_beta_sample(state, beta, action)
        
        # calcuate correction
        # proposed reinforce with off-policy-correction : policy / historical policy
        corr = torch.exp(phi_log_prob) / torch.exp(beta_log_prob)
        
        self.correction.append(corr)
        self.saved_log_probs.append(phi_log_prob)
        
        return phi_action, phi_probs

    def _select_action_with_TopK_correction(
        self, state, beta, action, K, **kwargs,
    ):
        phi_action, beta_action, phi_log_prob, beta_log_prob, phi_probs = self.phi_beta_sample(state, beta, action)
        
        # calcuate correction
        corr = torch.exp(phi_log_prob) / torch.exp(beta_log_prob)
        
        # cacluate Top K correction
        l_k = K * (1 - torch.exp(phi_log_prob)) ** (K - 1)
        
        self.correction.append(corr)
        self.lambda_k.append(l_k)
        self.saved_log_probs.append(phi_log_prob)
        
        return phi_action, phi_probs



class Critic(nn.Module):
    """Vanilla critic

    Args:
        nn (_type_): _description_

    Raises:
        NotImplemented: _description_

    Returns:
        _type_: _description_
    """
    def __init__(self, input_dim, action_dim, hidden_size, init_w=3e-5):
        super(Critic, self).__init__()
        
        self.drop_layer = nn.Dropout(p=0.5)
        
        self.linear1 = nn.Linear(input_dim + action_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        value = torch.cat([state, action], 1)
        value = F.relu(self.linear1(value))
        value = self.drop_layer(value)
        value = F.relu(self.linear2(value))
        value = self.drop_layer(value)
        value = self.linear3(value)
        return value

class ChooseREINFORCE():
    def __init__(self, method=None):
        if method is None:
            method = ChooseREINFORCE.basic_reinforce
        self.method = method
    
    @staticmethod
    def basic_reinforce(policy, returns, *args, **kwargs):
        policy_loss = []
        for log_prob, r in zip(policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * r)
        policy_loss = torch.cat(policy_loss).sum()
        return policy_loss
    
    @staticmethod
    def reinforce_with_correction(policy, returns, *args, **kwargs):
        policy_loss = []
        for corr, log_prob, R in zip(
            policy.correction, policy.saved_log_probs, returns
        ):
            policy_loss.append(corr * -log_prob * R) # <- this line here
        policy_loss = torch.cat(policy_loss).sum()
        return policy_loss
    
    @staticmethod
    def reinforce_with_TopK_correction(policy, returns, *args, **kwargs):
        policy_loss = []
        for l_k, corr, log_prob, R in zip(
            policy.lambda_k, policy.correction, policy.saved_log_probs, returns
        ):
            policy_loss.append(l_k * corr * -log_prob * R) # <- this line here
        
        policy_loss = torch.cat(policy_loss).sum()
        return policy_loss
    
    
    def __call__(self, policy, optimizer, learn=True):
        R = 0
        
        returns = []
        for r in policy.rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 0.0001)
        
        policy_loss = self.method(policy, returns)
        
        if learn: 
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()
            
        policy.gc()
        gc.collect()
        
        return policy_loss