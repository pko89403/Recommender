import numpy as np
import matplotlib.pyplot as plt 

from eps_bandit import model as eps
from ucb_bandit import model as ucb 
from eps_decay_bandit import model as eps_decay

if __name__ == '__main__':
    k = 10
    iters = 1000 
    episodes = 1000

    test_models = []
    rewards = []    

    test_models.append(ucb(k, 2, iters, 'random'))
    test_models.append(eps(k,0.1, iters, test_models[0].mu.copy()))
    test_models.append(eps_decay(k,None, iters, test_models[0].mu.copy()))

    plt.figure(figsize=(12,8))

    for idx, test_model in enumerate(test_models):
        rewards.append( np.zeros(iters) )
        for episode in range(episodes):
            test_model.reset('random')
            test_model.run()
            rewards[idx] += (test_model.reward - rewards[idx]) / ( episode + 1 )

        print(f"{test_model}, Mean Reward - {test_model.mean_reward}\n")
        plt.plot(rewards[idx], label=f"{test_model}-{test_model.param}")
        
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.title('Average Rewards for UCB and EPS')
    plt.savefig('AverageRewards.png')
    plt.close()

    width = 0.30
    bins = np.linspace(0, k-1, k) - width / len(test_models)
    plt.figure(figsize=(18,8))

    for idx, test_model in enumerate(test_models):
        plt.bar(bins + width * (idx-1), 
                test_model.k_n,
                width=width,
                label=f"{test_model}-{test_model.param}")
    plt.legend()
    plt.title("Number of Actions Selected by Each Algorithm")
    plt.xlabel("Action")
    plt.ylabel("Number of Actions Taken")
    plt.savefig('ActionTaken.png')
    plt.close()

