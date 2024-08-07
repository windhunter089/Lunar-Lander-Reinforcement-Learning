"""
Project 2: Lunar lander
tpham328
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

"""
1. Set up environment
2. Define DQN Architecture
3. Define Agent
4. Create function to train agent
5. Evaluate the trained agent
6. Hyperparameters tune
"""

# Define DQN architecture , feed-forward neural network
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

class DQN(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, hidden_dims):
        super(DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

        self.fc1 = nn.Linear(*self.state_dim, self.hidden_dims[0])
        # self.fc1_bn = nn.BatchNorm1d(self.hidden_dims[0])
        # self.fc1_ln = nn.LayerNorm(self.hidden_dims[0])
        self.fc2 = nn.Linear(self.hidden_dims[0], self.hidden_dims[1])
        # self.fc2_ln = nn.LayerNorm(self.hidden_dims[1])
        # self.fc2_bn = nn.BatchNorm1d(self.hidden_dims[1])
        self.fc3 = nn.Linear(self.hidden_dims[1], self.action_dim)
        # self.dropout = nn.Dropout(p = 0.1)
        
        self.optimizer = optim.Adam(self.parameters(), lr = alpha)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        # self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc1_ln(self.fc1(x)))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc2_ln(self.fc2(x)))

        return self.fc3(x)

# Define the agent
class Agent():
    def __init__(self, gamma, epsilon, alpha, state_dim, action_dim, batch_size,
                 eps_min, eps_decay):
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.action_list = [i for i in range(action_dim)]
        self.max_mem_size = 100000
        self.batch_size = batch_size
        self.mem_counter = 0

        # Q-network
        # Set up 2 different Q-network for online network and target network
        self.Q_network = DQN(self.alpha, action_dim = action_dim, state_dim= state_dim, hidden_dims = [128,128])
        self.Q_target = DQN(self.alpha, action_dim = action_dim, state_dim= state_dim, hidden_dims = [128,128])
        self.Q_target.load_state_dict(self.Q_network.state_dict())
        
        # Set up a buffer to store the experience (current state, action, reward, new state, terminal state)
        self.state_memory = np.zeros((self.max_mem_size, *state_dim), dtype = np.float32)
        self.action_memory = np.zeros(self.max_mem_size, dtype = np.int32)
        self.new_state_memory = np.zeros((self.max_mem_size, *state_dim), dtype = np.float32)
        self.reward_memory = np.zeros(self.max_mem_size, dtype = np.float32)
        self.terminal_memory = np.zeros(self.max_mem_size, dtype = np.bool_)

    def ChooseAction(self,observation):
        # random chance of exploration phase
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_list)
        # greedy pick
        else:
            state = T.tensor([observation]).to(self.Q_network.device)
            return T.argmax(self.Q_network.forward(state)).item()
    
    # function to update the Q target network
    def update_target_network(self):
        self.Q_target.load_state_dict(self.Q_network.state_dict())

    def train(self):
        if self.mem_counter >= self.batch_size:
            self.Q_network.optimizer.zero_grad()
            # in_batch is unique indeces of random sample from memory
            # current memory need to be more than batch_size that was sample
            batch_sample = np.random.choice(min(self.mem_counter, self.max_mem_size), self.batch_size, replace = False)

            # get sample experience
            state = T.tensor(self.state_memory[batch_sample]).to(self.Q_network.device)
            new_state = T.tensor(self.new_state_memory[batch_sample]).to(self.Q_network.device)
            reward = T.tensor(self.reward_memory[batch_sample]).to(self.Q_network.device)
            terminal = T.tensor(self.terminal_memory[batch_sample]).to(self.Q_network.device)
            action = self.action_memory[batch_sample]
            # print(type(terminal))

            # update Q value
            batch_index = np.arange(self.batch_size, dtype = np.int32)
            q_current = self.Q_network.forward(state)[batch_index, action]

            q_next = self.Q_target.forward(new_state)
            q_eval_next = self.Q_network.forward(new_state)
            q_next[terminal] = 0.0
            max_action = T.argmax(q_eval_next, dim = 1)
            q_target = reward + self.gamma * q_next.gather(1, max_action.unsqueeze(1)).squeeze(1)


            # q_target = reward + self.gamma * T.max(q_next, dim = 1)[0]
            loss = self.Q_network.loss(q_target, q_current).to(self.Q_network.device)
            loss.backward()
            self.Q_network.optimizer.step()

            #Decay epsilon
            if self.epsilon > self.eps_min:
                self.epsilon = self.epsilon - self.eps_decay
                # print('self.epsilon',self.epsilon)
            else:
                self.eps_min
            
if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma = 0.99, epsilon = 1.00, alpha = 0.0003, state_dim=[8],
                  action_dim = 4, batch_size = 64, eps_min= 0.01, eps_decay=0.001)
    reward_final = []
    train_episode = 600

    for i in range(train_episode):
        reward_episode = 0
        terminal = False
        observation = env.reset()
        upd_target_counter = 0
        while not terminal:
            # set pointer to go back overwrite oldest mem if full
            pointer = agent.mem_counter % agent.max_mem_size

            # experience / step
            action = agent.ChooseAction(observation)
            observation_prime, reward, terminal, _ = env.step(action)
            reward_episode += reward
            agent.state_memory[pointer] = observation
            agent.action_memory[pointer] = action
            agent.reward_memory[pointer] = reward
            agent.new_state_memory[pointer] = observation_prime
            agent.terminal_memory[pointer] = terminal
            agent.mem_counter += 1

            # train
            agent.train()
            # move to next time step
            observation = observation_prime
        
        # Compile reward and calculate average
        reward_final.append(reward_episode)
        reward_avg = np.mean(reward_final[-100:])

        # update Q target network
        upd_target_counter += 1
        if (upd_target_counter) == 1:
            agent.update_target_network()
            upd_target_counter = 0
        print('train episode', i+1, 'eps_reward ', reward_episode, 'avg reward ', reward_avg)

    #PLOT TRAIN RESULT 
    x1 = [i + 1 for i in range(train_episode)]
    reward_SMA1 = [np.mean(reward_final[max(0, i -100):i+1]) for i in range(train_episode)]
    # plt.plot(x, reward_final, 'o', color ='b', markersize = 3)
    plt.plot(x1, reward_final, color ='b', markersize = 3)
    plt.plot(x1, reward_SMA1, color = 'r', linewidth = 2)
    plt.title('reward at each episode')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.ylim(-500, 500)
    plt.grid(True)
    plt.savefig('p2.DDQN.png')
    # plt.show()
    plt.clf()
            
    # TEST 100 games with trained agent
    test_episode = 100
    test_reward = []
    for i in range(test_episode):
        reward_episode = 0
        terminal = False
        observation = env.reset()

        while not terminal:
            action = agent.ChooseAction(observation)
            observation_prime, reward, terminal, _ = env.step(action)
            reward_episode += reward
            observation = observation_prime
        
        test_reward.append(reward_episode)
        reward_avg = np.mean(test_reward[-100:])

        print('test episode', i+1, '/eps_reward ', reward_episode, '/avg reward', reward_avg)

    x2 = [i + 1 for i in range(test_episode)]
    reward_SMA2 = [np.mean(test_reward[max(0, i -100):i+1]) for i in range(test_episode)]
    # plt.plot(x2, test_reward, 'o', color ='b',markersize = 3)
    plt.plot(x2, test_reward, color ='b',markersize = 3)
    plt.plot(x2, reward_SMA2, color = 'r', linewidth = 2)
    plt.title('reward for 100 episode')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.ylim(-500, 500)
    plt.grid(True)
    plt.savefig('p2.testDDQN.png')
    # plt.show()
    plt.clf()