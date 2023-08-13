import numpy as np
from abc import ABCMeta

class Fabrication_Eval_Base(metaclass=ABCMeta):
    def __init__(self, I_index = [5,6,7,8], T_index = [9,10,11,12], X_index = [13]):
        self.base_score = 30
        self.I_index = I_index
        self.T_index = T_index
        self.X_index = X_index
        self.type_count = np.zeros(3)

    def count_type(self, new_type):
        if new_type in self.I_index:
             self.type_count[0] += 1
        elif new_type in self.T_index:
             self.type_count[1] += 1
        elif new_type in self.X_index:
             self.type_count[2] += 1
     
    def reset(self):
        self.type_count = np.zeros(3)  

