import gym
import gym_battleship

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import random
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttackerDQNnet(nn.Module):

    def __init__(self,action_space):
        super(AttackerDQNnet, self).__init__()
        self.cov1 = nn.Conv2d(2, 1, 1, padding="same")
        self.linear1 = nn.Linear(100, 200)
        self.linear2 = nn.Linear(200, 400)
        self.linear3 = nn.Linear(400, 200)
        self.linear4 = nn.Linear(200, action_space)
    
    def forward(self, state):
        o1 = F.relu(self.cov1(state))
        o2 = F.relu(self.linear1(torch.flatten(o1 , 1)))
        o3 = F.relu(self.linear2(o2))
        o4 = F.relu(self.linear3(o3))
        oo = self.linear4(o4)
        return oo

class AttackerDQN():
    def __init__(self, env, adversarial=False, test=False, REPLAY_SIZE=1000000, BATCH_SIZE=100, lr=.001, epsilon=.2, gamma=.99):
        self.BATCH_SIZE = BATCH_SIZE
        self.REPLAY_SIZE = REPLAY_SIZE

        self.lr = lr
        self.epsilon = epsilon
        self.env = env
        self.gamma = gamma
        
        self.test = test
        if adversarial:
            self.ACTION_SPACE = self.env.attacker_action_space.n
        else:
            self.ACTION_SPACE = self.env.action_space.n
        self.action_mask = np.zeros(self.ACTION_SPACE)

        
        self.target_net = AttackerDQNnet(self.ACTION_SPACE).to(device)
        self.optim = torch.optim.Adam(self.target_net.parameters(), lr=self.lr)
        
        self.replay_buffer = deque()
        self.loss_function = nn.MSELoss()

    def set_adversarial(self, flag):
        if flag:
            self.ACTION_SPACE = self.env.attacker_action_space.n
        else:
            self.ACTION_SPACE = self.env.action_space.n

    #Choose max unmasked action accroding to epsilon greedy policy
    def choose_action(self, ob):
        ob = torch.unsqueeze(torch.FloatTensor(ob).to(device), 0)

        if np.random.uniform() > self.epsilon:
            action_value = self.target_net(ob.to(device))
            ac = action_value.detach().cpu().numpy()
            #mask previously chosen actions
            masked = np.ma.masked_array(ac, self.action_mask)
            # print(masked)
            action = np.argmax(masked)
            #Set chosen action to be masked
            self.action_mask[action] = 1
        else:
            #Randomly pick a action that is not masked
            action = np.random.choice(np.argwhere(self.action_mask == 0).reshape(-1))
            self.action_mask[action] = 1
        return int(action)

    #Save the S.A.R.S experience to the replay buffer
    def save_transition(self, s, a, r, ns, done):
        one_hot_action = np.zeros(self.ACTION_SPACE)
        one_hot_action[a] = 1
        self.replay_buffer.append((s, one_hot_action, r, ns, done))
        if len(self.replay_buffer) > self.REPLAY_SIZE:
            self.replay_buffer.popleft()
        if len(self.replay_buffer) > self.BATCH_SIZE:
            self.learn()

    #Sample a batch from replay buffer and train the nn
    def learn(self):
        
        minibatch = random.sample(self.replay_buffer,self.BATCH_SIZE)
        state_batch = torch.FloatTensor([d[0] for d in minibatch]).to(device)
        action_batch = torch.LongTensor([d[1] for d in minibatch]).to(device)
        reward_batch = torch.FloatTensor([d[2] for d in minibatch]).to(device)
        nstate_batch = torch.FloatTensor([d[3] for d in minibatch]).to(device)
        done_batch = torch.FloatTensor([d[4] for d in minibatch]).to(device)
        done_batch = done_batch.unsqueeze(1)
        reward_batch = reward_batch.unsqueeze(1)
        action_index = action_batch.argmax(dim=1).unsqueeze(1)
        q = self.target_net(state_batch).gather(1, action_index)
        naction_batch = torch.unsqueeze(torch.max(self.target_net(nstate_batch), 1)[1], 1)
        next_q = self.target_net(nstate_batch).gather(1, naction_batch)

        delta = reward_batch + self.gamma * next_q * (1- done_batch)

        loss = self.loss_function(q, delta)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    #Run a training task on the agent. For pre-train purposes before going to adversarial game
    def run(self, episode=100,test=False):
        self.test = test
        
        avg_reward = []
        avg_step = []
        for e in tqdm( range(episode) ):
            r = 0
            state = self.env.reset()
            self.action_mask = np.zeros(self.ACTION_SPACE)
            step = 0
            p_reward = 0
            # env.render_board_generated()
            while True:
                step += 1
                action = self.choose_action(state)
                
                nstate, reward, done, _ = self.env.step(action)
                #Modify the reward, makes the hit reward accumlative
                if p_reward > 0 and reward > 0:
                    reward += p_reward
                if test:
                    self.env.render()
                    print(action)
                r += reward
                if not self.test:
                    self.save_transition(state, action, reward, nstate, done)
                state = nstate
                p_reward = reward

                if done:
                    break
            avg_reward.append(r)
            avg_step.append(step)
        return avg_reward, avg_step




class Actor(nn.Module):

    def __init__(self, action_space):
        super(Actor, self).__init__()
        self.cov1 = nn.Conv2d(2, 1, 1, padding="same")
        self.linear1 = nn.Linear(100, 200)
        self.linear2 = nn.Linear(200, 400)
        self.linear3 = nn.Linear(400, 200)
        self.linear4 = nn.Linear(200, action_space)

    def forward(self, state):
        o1 = F.relu(self.cov1(state))
        o2 = F.relu(self.linear1(torch.flatten(o1 , 1)))
        o3 = F.relu(self.linear2(o2))
        o4 = F.relu(self.linear3(o3))
        oo = self.linear4(o4)
        #We need to do masking, so not return distribution here
        return oo


class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        self.cov1 = nn.Conv2d(2, 1, 1, padding="same")
        self.linear1 = nn.Linear(100, 200)
        self.linear2 = nn.Linear(200, 400)
        self.linear3 = nn.Linear(400, 200)
        self.linear4 = nn.Linear(200, 1)

    def forward(self, state):
        o1 = F.relu(self.cov1(state))
        o2 = F.relu(self.linear1(torch.flatten(o1 , 1)))
        o3 = F.relu(self.linear2(o2))
        o4 = F.relu(self.linear3(o3))
        oo = self.linear4(o4)
        return oo

#The following implementation is similar to assignment5 except the action masking part.
class ActorCritic_Batch():

    def __init__(self, env, adversarial=False, gamma=.9, actor_lr=.001, critic_lr=.001):
        self.env = env
        if adversarial:
            self.ACTION_SPACE = self.env.attacker_action_space.n
        else:
            self.ACTION_SPACE = self.env.action_space.n   
        self.actor = Actor(self.ACTION_SPACE).to(device)
        self.critic = Critic().to(device)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        
        self.gamma = gamma

        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr = self.critic_lr)
        self.action_mask = torch.zeros(self.ACTION_SPACE, dtype=torch.bool).to(device)
    
    def choose_action(self, ob):

        action_value = self.actor(ob)
        #To mask a tensor, I set the value of the masked action to be low, and it is almost impossile to be chosen
        masked = action_value.masked_fill(self.action_mask, -99999999)
        dist = Categorical(F.softmax(masked, dim=-1))
        action = dist.sample()

        self.action_mask[action] = True
        return action, Categorical(F.softmax(action_value, dim=-1))



    def update(self, actor_loss, critic_loss):

        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()
        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()

    def compute_returns(self, next_value, rewards, masks):

        rewards = torch.cat(rewards)
        masks = torch.cat(masks)

        gamma = 1
        returns = 0
        for i in range(next_value, len(rewards)):
            gamma = gamma * self.gamma
        returns += gamma * (rewards[i] * masks[i])
        return returns


    def run(self, episode=100, test=False):

        episode_rewards = []
        steps = []

        
        for e in tqdm( range(episode) ):
            log_probs = []
            values = []
            rewards = []
            masks = []

            state = self.env.reset()
            self.action_mask = torch.zeros(self.ACTION_SPACE, dtype=torch.bool).to(device)
            state = torch.FloatTensor(state).to(device)
            step = 0
            r = 0
            p_reward = 0
            while True:
                step += 1
                action_value = self.actor(state)

                masked = action_value.masked_fill(self.action_mask, -99999999)
                masked_dist = Categorical(F.softmax(masked, dim=-1))
                action = masked_dist.sample()
                self.action_mask[action] = True

                dist = Categorical(F.softmax(action_value, dim=-1))
                value = self.critic(state)

                next_state, reward, done, _ = self.env.step(action.item()) 
                if p_reward > 0 and reward > 0:
                    reward += p_reward
                r += reward

                next_state = torch.FloatTensor(next_state).to(device)
                next_value = self.critic(next_state)
                log_prob = dist.log_prob(action).unsqueeze(0)
                
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
                masks.append(torch.tensor([1-done], dtype=torch.float, device=device))
                if done:
                    break

                state = next_state

            delta = []
            for i in range(0, len(values)):
                delta.append(self.compute_returns(i, rewards, masks).unsqueeze(0) - values[i])
            # print(delta)
            critic_loss = torch.mean(torch.cat(delta) ** 2)
            actor_loss = torch.mean(torch.cat(delta).detach() * (- torch.cat(log_probs)))     

            self.update(actor_loss, critic_loss)

            steps.append(step)
            episode_rewards.append(r)

        return episode_rewards, steps


def sample_run():
    env = gym.make('Battleship-v0', reward_dictionary=
    {'win':0, 'missed': -10, 'touched': 5})
    a1 = AttackerDQN(env)
    a1.target_net.load_state_dict(torch.load('./100kR100Br(0,-10,5,+p)c1L3_onehot'))
    r1, s1 = a1.run(episode=1, test=True)
    plt.plot(r1)
    plt.show()
    plt.plot(s1)
    plt.show()

if __name__ == "__main__":
    env = gym.make('Battleship-v0', reward_dictionary=
    {'win':0, 'missed': -10, 'touched': 5})
    # a1 = ActorCritic_Batch(env)
    # state = env.reset()
    # r1, s1 = a1.run(episode=2000)

    # print(sum(s1) / len(s1))

    # #Save the network weight
    # torch.save(a1.actor.state_dict(), "./actor")
    # torch.save(a1.critic.state_dict(), "./critic")

    a2 = AttackerDQN(env, epsilon=.2)
    state = env.reset()
    r2, s2 = a2.run(episode=5000)

    print(sum(s2) / len(s2))
    # plt.plot(r1)
    plt.plot(r2)
    # plt.show()

    # plt.plot(s1)
    plt.plot(s2)
    plt.show()
    #Save the network weight
    torch.save(a2.target_net.state_dict(), "./aDQN")