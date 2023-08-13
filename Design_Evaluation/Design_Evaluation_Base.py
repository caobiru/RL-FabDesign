import numpy as np
from abc import ABCMeta

class Design_Eval_Base(metaclass=ABCMeta):
    def __init__(self, room_types = 5):
        self.base_score = 50
        self.type_count = np.zeros(room_types)
        self.type_list = [i for i in range(room_types)]
        self.room_types = room_types

    def count_type(self, new_type):
        if new_type in self.type_list:
             self.type_count[new_type] += 1

    def reset(self):
        self.type_count = np.zeros(self.room_types)


