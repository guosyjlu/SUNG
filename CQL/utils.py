import random
import pprint
import time
import uuid
import tempfile
import os
from copy import copy
from socket import gethostname
import pickle

import numpy as np

import absl.flags
from absl import logging
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags
from ml_collections.config_dict import config_dict

import wandb
import torch


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
        self.p = p
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def add(self, state, action, next_state, reward, done):
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
        self.mean = self.offline_buffer.state.mean(0, keepdims=True)
        self.std = self.offline_buffer.state.std(0, keepdims=True) + 1e-3
        self.mean = torch.tensor(self.mean).to(self.device)
        self.std = torch.tensor(self.std).to(self.device)
        self.offline_buffer.restore_buffer()


def define_flags_with_default(**kwargs):
    for key, val in kwargs.items():
        if isinstance(val, ConfigDict):
            config_flags.DEFINE_config_dict(key, val)
        elif isinstance(val, bool):
            # Note that True and False are instances of int.
            absl.flags.DEFINE_bool(key, val, 'automatically defined flag')
        elif isinstance(val, int):
            absl.flags.DEFINE_integer(key, val, 'automatically defined flag')
        elif isinstance(val, float):
            absl.flags.DEFINE_float(key, val, 'automatically defined flag')
        elif isinstance(val, str):
            absl.flags.DEFINE_string(key, val, 'automatically defined flag')
        else:
            raise ValueError('Incorrect value type')
    return kwargs


def set_random_seed(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)



def get_user_flags(flags, flags_def):
    output = {}
    for key in flags_def:
        val = getattr(flags, key)
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            output[key] = val

    return output


def flatten_config_dict(config, prefix=None):
    output = {}
    for key, val in config.items():
        if prefix is not None:
            next_prefix = '{}.{}'.format(prefix, key)
        else:
            next_prefix = key
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=next_prefix))
        else:
            output[next_prefix] = val
    return output



def prefix_metrics(metrics, prefix):
    return {
        '{}/{}'.format(prefix, key): value for key, value in metrics.items()
    }
