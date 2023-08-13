from .DQN_Base import DuelingDeepQNetwork,ReplayBuffer
from abc import ABCMeta
import numpy as np
from tensorflow.keras.models import load_model
from Team_Project.Behaviour_Function import mask_actions

class Agent_Base(metaclass=ABCMeta):
    
    def __init__(self, n_actions=16, batch_size=64,input_dims=2,epsilon=1.0,lr=0.0005, gamma=0.97, epsilon_dec=1e-4, eps_end=0.03,
                 mem_size=100000, fc1_dims=512,
                 fc2_dims=256, replace=100,**kwargs):
        self.action_num = n_actions
        self.actions_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = eps_end
        self.replace = replace
        self.batch_size = batch_size
        self.learn_step_counter = 0
        self.input_dims = input_dims
        self.mem_size = mem_size
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lr = lr
        self.range = np.arange(self.batch_size, dtype=np.int32)
        # self.q_eval = DuelingDeepQNetwork(n_actions, self.fc1_dims, self.fc2_dims)
        # self.q_next = DuelingDeepQNetwork(n_actions, self.fc1_dims, self.fc2_dims)
        self.q_eval = None
        self.q_next = None
        self.memory = ReplayBuffer(self.mem_size, input_dims,self.action_num)

    def store_transition(self, state, action, reward, new_state, mask, done):
        self.memory.store_transition(state, action, reward, new_state, mask, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        if self.learn_step_counter % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

        states, actions, rewards, states_, masks, dones = \
            self.memory.sample_buffer(self.batch_size)

        q_pred = self.q_eval.predict(states, verbose=0)
        q_next = self.q_eval.predict(states_ ,verbose=0)
        q_target = self.q_next.predict(states_, verbose=0)

        # q_target = q_pred.numpy()
        max_actions_masked = mask_actions(q_next,self.actions_space,self.action_num,masks,self.batch_size)

        # for idx, terminal in enumerate(dones): 
        #         q_target[idx, actions[idx]] = rewards[idx] + \
        #                                       self.gamma * q_target[idx, max_actions_masked[idx]] * (1 - int(dones[idx]))

        q_pred[self.range, actions] = rewards + self.gamma*q_target[self.range, max_actions_masked]*(1-dones)

        self.q_eval.fit(states, q_pred, verbose=0)#, callbacks=[tf_callback]
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
                                                      self.eps_min else self.eps_min
        self.learn_step_counter += 1

    def save_model(self, path):
        self.q_eval.save(path)

    def load_model(self, path):
        self.q_eval = load_model(path)


