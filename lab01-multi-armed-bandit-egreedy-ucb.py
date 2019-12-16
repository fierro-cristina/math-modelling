import numpy as np
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, k = 10, exploration_rate = 0.3, learning_rate = 0.1, ucb = False, seed = None, c = 2):
        self.k = k #number of agents initialized
        self.actions = range(self.k)
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.total_reward = 0
        self.average_reward = []

        self.end_value = []
        np.random.seed(seed)
        #The reward is randomly generated from the normal distribution.
        for i in range(self.k):
            self.end_value.append(np.random.randn() + 2) #normal distribution

        self.values = np.zeros(self.k)
        self.times = 0
        self.action_times = np.zeros(self.k)

        self.ucb = ucb
        self.c = c

#Choosing actions condenses down to two methods: e-greedy and ucb (Upper Confodence Bound)
    def chooseAction(self):
        #Exploration
        if np.random.uniform(0,1) <= self.exploration_rate:
            action = np.random.choice(self.actions)
        else:
        #Exploitation
            if self.ucb:
                if self.times == 0:
                    action = np.random.choice(self.actions)
                else:
                    confidence_bound = self.values + self.c * np.sqrt(np.log(self.times) / (self.action_times + 0.1))
                    action = np.argmax(confidence_bound)
            else:
                action = np.argmax(self.values)
        return action

    def takeAction(self, action):
        self.times += 1
        self.action_times[action] += 1

        reward = np.random.randn() + self.end_value[action]

        self.values[action] += self.learning_rate * (reward - self.values[action])

        self.total_reward += reward
        self.average_reward.append(self.total_reward/self.times)

    def Act(self, n):
        for _ in range(n):
            action = self.chooseAction()
            self.takeAction(action)

if __name__ == "__main__":
    agent = Agent(k = 5, exploration_rate = 0, seed = 1234) #e-greedy
    agent.Act(2000)

    print("Estimated Values: ", agent.values)
    print("Actual Values: ", agent.end_value)

    average_reward_1 = agent.average_reward

    agent = Agent(k = 5, exploration_rate = 0.01, seed = 1234) #e-greedy
    agent.Act(2000)

    print("Estimated Values: ", agent.values)
    print("Actual Values: ", agent.end_value)

    average_reward_2 = agent.average_reward

    agent = Agent(k = 5, exploration_rate = 0.1, seed = 1234) #e-greedy
    agent.Act(2000)

    print("Estimated Values: ", agent.values)
    print("Actual Values: ", agent.end_value)

    average_reward_3 = agent.average_reward

    agent = Agent(k = 5, exploration_rate = 0.4, seed = 1234) #e-greedy
    agent.Act(2000)

    print("Estimated Values: ", agent.values)
    print("Actual Values: ", agent.end_value)

    average_reward_4 = agent.average_reward

    agent = Agent(k = 5, exploration_rate = 0.1, seed = 1234, ucb = True, c = 1)
    agent.Act(2000)

    print("Estimated Values: ", agent.values)
    print("Actual Values: ", agent.end_value)

    average_reward_5 = agent.average_reward

    agent = Agent(k = 5, exploration_rate = 0.1, seed = 1234, ucb = True, c = 2)
    agent.Act(2000)

    print("Estimated Values: ", agent.values)
    print("Actual Values: ", agent.end_value)

    average_reward_6 = agent.average_reward

    agent = Agent(k = 5, exploration_rate = 0.1, seed = 1234, ucb = True, c = 10)
    agent.Act(2000)

    print("Estimated Values: ", agent.values)
    print("Actual Values: ", agent.end_value)

    average_reward_7 = agent.average_reward


    #Rewards plot
    plt.figure(figsize = [8, 6])
    plt.plot(average_reward_1, label = "e = 0")
    plt.plot(average_reward_2, label = "e = 0.01")
    plt.plot(average_reward_3, label = "e = 0.1")
    plt.plot(average_reward_4, label = "e = 0.4")
    plt.plot(average_reward_5, label = "UCB, c = 1")
    plt.plot(average_reward_6, label = "UCB, c = 2")
    plt.plot(average_reward_7, label = "UCB, c = 10")
    plt.xlabel("Actions", fontsize = 10
    plt.ylabel("Average Reward", fontsize = 10
    plt.show()
