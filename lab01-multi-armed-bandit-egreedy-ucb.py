import numpy as np
import matplotlib.pyplot as plt

arms_num = 10
expected = np.random.randint(0, arms_num, arms_num)
variance = np.around(np.random.rand(arms_num), decimals = 4)
std = np.sqrt(variance)
arms_rewards = np.array([np.random.normal(expected[i], std[i], 1)[0] for i in range(arms_num)])
arms_mu = np.around(arms_rewards, decimals = 3)
epsilon = np.array([0, 0.1, 0.01, 0.4])
Q_0 = np.array([1, 1.5, 2, 5])
c = np.array([1, 2, 10])

class Agent:
    def __init__(self, **kwargs):
        for i, j in kwargs.items():
            setattr(self, i, j)

        self.k = 10
        self.n = 1
        self.k_n = np.ones(self.k)
        self.mean_reward = 0
        self.reward = np.zeros(self.iters)
        self.k_reward = np.zeros(self.k)
        self.mu = self.calcMu()

    def calcMu(self):
        expected = np.random.randint(0, self.k, self.k)
        variance = np.around(np.random.rand(self.k), decimals = 4)
        std = np.sqrt(variance)
        arms_rewards = np.array([np.random.normal(expected[i], std[i], 1)[0] for i in range(self.k)])
        arms_mu = np.around(arms_rewards, decimals = 3)
        self.q_max = std[np.argmax(arms_mu)] + max(arms_mu)

        return arms_mu

    def Reward(self):
        i = int(self.current_arm)
        if i == len(self.mu):
            print('current arm: %s\n%s'%(i, self.current_arm))
        reward = np.random.normal(self.mu[i], 1)

        self.n += 1
        self.k_n[i] += 1
        self.mean_reward = self.mean_reward + (reward - self.mean_reward)/self.n
        self.k_reward[i] = self.k_reward[i] + (reward - self.k_reward[i])/self.k_n[i]

    def Play(self):
        for i in range(self.iters):
            self.select_arm()
            self.Reward()
            self.reward[i] = self.mean_reward

    def Reset(self):
        self.n = 1
        self.k_n = np.ones(self.k)
        self.mean_reward = 0
        self.reward = np.zeros(self.iters)
        self.k_reward = np.zeros(self.k)
        self.mu = self.calcMu()

class e_greedy(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def select_arm(self):
        if np.random.rand() > self.eps:
            self.current_arm = self.k_reward.argmax()
        else:
            self.current_arm = np.random.choice(range(len(self.k_reward)))

class Optimistic(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reward.fill(self.q_0)

    def select_arm(self):
        self.current_arm = self.k_reward.argmax()

class UCB(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def select_arm(self):
        self.current_arm = np.argmax(self.k_reward + self.c * np.sqrt((np.log(self.n))/self.k_n))

def Start(params, alg = 'epsilon', mu = arms_mu, iters = 1000, episodes = 1000):
    if alg == 'e-greedy':
        stats = {}
        print('-'*40)
        print('Method: e-greedy')
        for i in params['eps']:
            greed_rewards = np.zeros(iters)
            greed = e_greedy(eps = i, mu = arms_mu, iters = iters)

            for j in range(episodes):
                greed.Play()
                greed_rewards = greed_rewards + (greed.reward - greed_rewards) / (j + 1)
                greed.Reset()

            stats[i] = greed_rewards
            print('epsilon:', i)
        plot_agents(alg, params['eps'], stats, episodes)
    elif alg == 'Optimistic':
        stats = {}
        leg = []
        print('-'*40)
        print('Method: Optimistic')
        for i in [1, 1.5, 2, 5]:
            optimistic_rewards = np.zeros(iters)
            optimistic = Optimistic(q_0 = i, mu = arms_mu, iters = iters)
            optimistic.q_0 = optimistic.q_max*i

            for j in range(episodes):
                optimistic.Play()
                optimistic_rewards = optimistic_rewards + (optimistic.reward - optimistic_rewards)/(j+1)
                optimistic.Reset()

            stats[optimistic.q_0] = optimistic_rewards
            print('Q_0: %.3f' % (optimistic.q_0))
            leg.append(optimistic.q_0)

        plot_agents(alg, leg, stats, episodes)
    elif alg == 'UCB':
        stats = {}
        print('-'*40)
        print('Method: UCB')
        for i in params['c']:
            ucb_rewards = np.zeros(iters)
            ucb = UCB(c = i, mu = arms_mu, iters = iters)

            for j in range(episodes):
                ucb.Play()
                ucb_rewards = ucb_rewards + (ucb.reward - ucb_rewards)/(j + 1)

                ucb.Reset()

            stats[i] = ucb_rewards
            print('c: ', i)
        plot_agents(alg, params['c'], stats, episodes)

def plot_agents(name, param, stats, episodes):
    fig = plt.figure(figsize = (16, 10))
    ax = plt.subplot(111)
    for i in param:
        ax.plot(stats[i], label = str(i))
    ax.legend(loc = 'lower right')
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Rewards")
    ax.set_title("Method: " + str(name))
    plt.show()

it = 1000
ep = 1000

parameters_dict = {'eps': epsilon, 'q_0':Q_0, 'c':c}

Start(parameters_dict, alg = 'e-greedy', iters = it, episodes = ep)
Start(parameters_dict, alg = 'Optimistic', iters = it, episodes = ep)
Start(parameters_dict, alg = 'UCB', iters = it, episodes = ep)
