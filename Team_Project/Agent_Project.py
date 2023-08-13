import RL_Base.Agent_Base as AgentBase
import RL_Base.DQN_Base as DQNBase
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import tensorflow as tf
import datetime

import platform
os_name = platform.system()
if os_name == 'Windows':
    # import Windows-specific modules
    from tensorflow.keras.optimizers import Adam
elif os_name == 'Darwin':
    # import MacOS-specific modules
    from tensorflow.keras.optimizers.legacy import Adam
else:
    print(f'Unsupported operating system: {os_name}')

import numpy as np

class Agent(AgentBase.Agent_Base):

    def __init__(self,n_actions,batch_size,input_dims,**kwargs):
        super(Agent, self).__init__(n_actions,batch_size,input_dims,**kwargs)
        self.action_num = n_actions
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.action_space = np.array([i for i in range(n_actions)])

    def update_parameters(self):
        self.memory = DQNBase.ReplayBuffer(self.mem_size, self.input_dims, self.action_num)
        nn = Sequential([Dense(256, input_shape=(self.input_dims,)),Activation('relu'),Dense(256),Activation('relu'),Dense(self.action_num)])
        nn.compile(optimizer=Adam(lr=self.lr), loss='mse',metrics=['accuracy'])
        self.q_eval = nn
        nnt = Sequential([Dense(256, input_shape=(self.input_dims,)),Activation('relu'),Dense(256),Activation('relu'),Dense(self.action_num)])
        nnt.compile(optimizer=Adam(lr=self.lr), loss='mse')
        self.q_next = nnt

    def choose_action_mask(self, observation, mask):
        avaliable_action = self.action_space[mask]
        if mask.shape == (1,):
            return int(mask)
        else:
            if np.random.random() < self.epsilon:
                action = np.random.choice(avaliable_action)
            else:
                actions = self.q_eval.predict(np.array([observation]), verbose=0)
                mask_out = np.setdiff1d(self.action_space, mask, assume_unique=False)
                actions_masked = actions.reshape(self.action_num)
                actions_masked[mask_out] = -999
                action = np.argmax(actions_masked)
            return action
        
    def choose_action(self, observation):

        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(np.array([observation]), verbose=0)
            action = np.argmax(actions)
        return action 

    def inference_action(self, observation, mask):
        if mask.shape == (1,):
            return int(mask)
        else:
            actions = self.q_eval.predict(np.array([observation]), verbose=0)
            mask_out = np.setdiff1d(self.action_space, mask, assume_unique=False)
            actions_masked = actions.reshape(self.action_num)
            actions_masked[mask_out] = -999
            action = np.argmax(actions_masked)
            return action