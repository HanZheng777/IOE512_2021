from collections import deque
import random
import numpy as np
# import pandas as pd
import math
from Learning import configs


def sample_n_unique(sampling_func, n):
    """
    Helper function. Given a function `sampling_func` that returns
    comparable objects, sample n such unique objects.
    :param n: (int) number of unique objects
    :return: (list) indexs of these objects
    """
    res = []
    while len(res) < n:
        candidate = sampling_func()
        if candidate not in res:
            res.append(candidate)
    return res


class ReplayBuffer(object):
    """
    Replay Buffer for Value Iteration
    """

    def __init__(self, size, horizon):
        """
        :param size: (int) buffer size
        :param horizon: (int) horizon of MDP
        """

        self.size = size
        self.data = {}
        self.horizon = horizon
        self.config = configs

        base = deque(maxlen=self.size)

        self.data['state'] = base.copy()
        self.data['action'] = base.copy()
        self.data['reward'] = base.copy()
        self.data['next_state'] = base.copy()
        self.data['done'] = base.copy()

    def can_sample(self, buffer_data_size, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size <= buffer_data_size

    def store(self,state,action, reward,next_state,done):
        """
        Stores the transition and virtual experiment results
        """

        self.data['state'].append(state)
        self.data['action'].append(action)
        self.data['reward'].append(reward)
        self.data['next_state'].append(next_state)
        self.data['done'].append(done)

    def encode_sample(self, idexs):
        """
        Helper function. Encodes batch episodes to corresponding transition data
        :param idexs: (list) sample indexs
        :return: (list) encoded mini batch data
        """
        return_list = []
        key_list = ['state', 'action', 'reward', 'next_state', 'done']
        for key in key_list:
            return_list.append(np.array(self.data[key])[idexs])

        return return_list

    def sample(self, batch_size):
        """
        Samples mini batch data
        :param batch_size: (int) batch size
        :return: (list) encoded mini batch data
        """
        num_in_buffer = len(self.data['state'])
        assert self.can_sample(num_in_buffer, batch_size)
        idexs = sample_n_unique(lambda: random.randint(0, num_in_buffer-1), batch_size)

        return self.encode_sample(idexs)