import Design_Evaluation.Design_Evaluation_Base as DesignEvalModel
import Fabrication_Evaluation.Fabrication_Evaluation_Base as FabEvalModel
from .Behaviour_Function import check_distance,map_reward
import numpy as np

class Design_Eval(DesignEvalModel.Design_Eval_Base):
    def __init__(self, room_types,**kwargs):
        super(Design_Eval, self).__init__(room_types,**kwargs)
        self.room_types = room_types
        self.type_count = np.zeros(room_types)
        self.type_list = [i for i in range(room_types)]
        self.re_range = [-75,30]
    
    def compute_design_reward(self,stair_exist,csv):
        if stair_exist == 1:
            loss_stair = 20
        else:
            loss_stair = -20
        loss_0 = check_distance(self.type_count[0],csv[0])
        loss_1 = check_distance(self.type_count[1],csv[1])
        loss_2 = check_distance(self.type_count[2],csv[2])
        loss_3 = check_distance(self.type_count[3],csv[3])

        final_reward = self.base_score + loss_0 + loss_1 + loss_2 + loss_3 + loss_stair

        return final_reward
    
    def compute_single_design_reward(self,new_type):
        if new_type in self.type_list:
            design_reward = 5
        else:
            design_reward = -5
        return design_reward*0.1

class Fabrication_Eval(FabEvalModel.Fabrication_Eval_Base):
    def __init__(self, I_index, T_index, X_index,Time_goal,**kwargs):
        super(Fabrication_Eval, self).__init__(I_index, T_index, X_index,**kwargs)
        self.I_index = I_index
        self.T_index = T_index
        self.X_index = X_index
        self.Time_goal = Time_goal
        self.re_range = [Time_goal-100,10]

    def compute_fabrication_reward(self,stair_exist,csv):
        if stair_exist == 1:
            time_stair = 10
        else:
            time_stair = 0
        I_fab_time = self.type_count[0]*csv[0]
        T_fab_time = self.type_count[1]*csv[1]
        X_fab_time = self.type_count[2]*csv[2]

        time_cost = time_stair + I_fab_time + T_fab_time + X_fab_time

        final_reward = self.base_score + min(0,(self.Time_goal - time_cost))

        return final_reward, time_cost
    
    def compute_single_fabrication_reward(self,new_type,csv):
        if new_type in self.I_index:
            time_reward = -csv[0]
        elif new_type in self.T_index:
            time_reward = -csv[1]
        elif new_type in self.X_index:
            time_reward = -csv[2]
        else:
            time_reward = 1

        return time_reward*0.1
