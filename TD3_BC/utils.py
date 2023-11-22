"""
	Offline-to-Online Replay Buffer: Online Buffer (20K, first-in-first-out) and Offline Buffer (3M).
	Sampling: p% ~ Online Buffer (minimum), and 1-p% ~ Offline Buffer.
	Default Setting: p = 10
"""
import numpy as np
import torch
import random

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		self.state_dim = state_dim
		self.action_dim = action_dim

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


	def convert_D4RL(self, dataset):
		self.state = dataset['observations']
		self.action = dataset['actions']
		self.next_state = dataset['next_observations']
		self.reward = dataset['rewards'].reshape(-1,1)
		self.not_done = 1. - dataset['terminals'].reshape(-1,1)
		self.size = self.state.shape[0]
		self.ptr = self.size


	def normalize_states(self, eps = 1e-3):
		mean = self.state.mean(0,keepdims=True)
		std = self.state.std(0,keepdims=True) + eps
		self.state = (self.state - mean)/std
		self.next_state = (self.next_state - mean)/std
		return mean, std

	def restore_buffer(self):
		size = self.max_size - self.size
		state = np.zeros((size, self.state_dim))
		action = np.zeros((size, self.action_dim))
		next_state = np.zeros((size, self.state_dim))
		reward = np.zeros((size, 1))
		not_done = np.zeros((size, 1))
		self.state = np.append(self.state, state, axis=0)
		self.action = np.append(self.action, action, axis=0)
		self.next_state = np.append(self.next_state, next_state, axis=0)
		self.reward = np.append(self.reward, reward, axis=0)
		self.not_done = np.append(self.not_done, not_done, axis=0)


class OORB(object):
    def __init__(self, state_dim, action_dim, offline_size=int(3e6), online_size=int(2e4), p=0.1):
        self.offline_buffer = ReplayBuffer(state_dim, action_dim, offline_size)
        self.online_buffer = ReplayBuffer(state_dim, action_dim, online_size)
        self.state_mean = None
        self.state_std = None
        self.p = p
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def add(self, state, action, next_state, reward, done):
        state = (state - self.state_mean) / self.state_std
        next_state = (next_state - self.state_mean) / self.state_std
        self.offline_buffer.add(state, action, next_state, reward, done)
        self.online_buffer.add(state, action, next_state, reward, done)
    
    def sample(self, batch_size):
        online_num = min(int(batch_size * (1 -self.p)), self.online_buffer.size)
        offline_num = batch_size - online_num
        online_ind = np.random.randint(0, self.online_buffer.size, size=online_num)
        offline_ind = np.random.randint(0, self.offline_buffer.size, size=offline_num)
        return (
			torch.FloatTensor(np.append(self.offline_buffer.state[offline_ind], self.online_buffer.state[online_ind], axis=0)).to(self.device),
			torch.FloatTensor(np.append(self.offline_buffer.action[offline_ind], self.online_buffer.action[online_ind], axis=0)).to(self.device),
			torch.FloatTensor(np.append(self.offline_buffer.next_state[offline_ind], self.online_buffer.next_state[online_ind], axis=0)).to(self.device),
			torch.FloatTensor(np.append(self.offline_buffer.reward[offline_ind], self.online_buffer.reward[online_ind], axis=0)).to(self.device),
			torch.FloatTensor(np.append(self.offline_buffer.not_done[offline_ind], self.online_buffer.not_done[online_ind], axis=0)).to(self.device)
		)
        
    def convert_D4RL(self, dataset):
        self.offline_buffer.convert_D4RL(dataset)
        self.state_mean, self.state_std = self.offline_buffer.normalize_states()
        self.offline_buffer.restore_buffer()
        

if __name__ == '__main__':
    import d4rl
    import gym
    env = gym.make("hopper-random-v2")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])
    replay_buffer = OORB(state_dim, action_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    # print(replay_buffer.state_mean)
    replay_buffer.add([-1]*11, [-1]*3, [-1]*11, 1, 1)
    replay_buffer.add([-1]*11, [-1]*3, [-1]*11, 1, 1)
    print(replay_buffer.sample(10))