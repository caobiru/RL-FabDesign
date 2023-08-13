import RL_Base.ENV_Base as BaseENVModel
import numpy as np
from .Behaviour_Function import checkleft,checkdown,checkleftdown,mapping_types
from .Reward_Project import Design_Eval,Fabrication_Eval

class EVN(BaseENVModel.EVN_Base):

    def __init__(self, n_actions, input_dims, room_types, n_x, n_y, rule_map, ind_stair, ind_I, ind_T, ind_X, time_goal,**kwargs):
        super(EVN, self).__init__(n_x,n_y,**kwargs)
        self.action_space = [i for i in range(n_actions)]
        self.action_num = n_actions
        self.input_dims = input_dims
        self.room_types = room_types
        self.nx = n_x
        self.ny = n_y
        self.sates_len = self.nx * self.ny
        self.state_record = np.zeros(self.sates_len)
        self.rulemap = rule_map
        self.stair_used = 0
        self.stair_left = 999
        self.ind_stair = ind_stair
        self.coverd_stair = []
        self.delet_ind = []
        self.eval_design = Design_Eval(room_types)
        self.eval_fab = Fabrication_Eval(ind_I, ind_T, ind_X, time_goal)

    def reset_project(self):
        self.stair_used = 0
        self.coverd_stair = []
        self.delet_ind = []
        self.stair_left = 999
        self.state_record = np.zeros(self.all_state, dtype=np.int32)
        self.eval_design.reset()
        self.eval_fab.reset()
        return np.ones(self.input_dims) #* (self.action_num + 1)

    def get_rule_to_mask(self, current_location):
        if current_location in self.coverd_stair:
            return np.array([4])
        if current_location == self.stair_left and current_location % self.nx !=self.nx-1:
            return np.array([7])
        else:
            x_loc = current_location % self.nx
            y_loc = current_location // self.nx

            if y_loc == 0:
                if x_loc == 0:
                    return np.array(self.action_space)
                else:
                    left_state = self.state_record[current_location - 1]
                    return checkleft(left_state, self.rulemap, self.delet_ind)
            elif x_loc == 0:
                down_state = self.state_record[current_location - self.nx]
                return checkdown(down_state, self.rulemap, self.delet_ind)
            else:
                left_state = self.state_record[current_location - 1]
                down_state = self.state_record[current_location - self.nx]
                return checkleftdown(left_state, down_state, self.rulemap, self.delet_ind)

    def step(self, action, current_location,csv_fab):

        x_loc = current_location % self.nx
        y_loc = current_location // self.nx
        # action = mapping_types(current_location, self.room_types, self.state_record, self.nx, action)

        if action == self.ind_stair[0] and self.stair_used == 0:
            self.stair_used = 1
            self.delet_ind.extend(self.ind_stair)
            self.coverd_stair = [current_location + self.nx - 1, current_location - 1]
            self.stair_left = current_location + self.nx - 2
        if action == self.ind_stair[1] and self.stair_used == 0:
            self.stair_used = 1
            self.delet_ind.extend(self.ind_stair)
            self.stair_left = current_location + self.nx - 2
            self.coverd_stair = [current_location + self.nx - 1, current_location - 1]
            
        self.state_record[current_location - 1] = int(action)

        self.eval_design.count_type(action)
        self.eval_fab.count_type(action)
        # print(self.state_record.tolist())
        reward = self.eval_fab.compute_single_fabrication_reward(action,csv_fab) + self.eval_design.compute_single_design_reward(action)
        mask_ind = self.get_rule_to_mask(current_location)
        mask_new = np.zeros(self.action_num, dtype=np.int32)
        mask_new[mask_ind] = 1

        if y_loc == 0:
            state_current = np.array([current_location/self.sates_len, 1, action/self.action_num])
        elif x_loc == 0:
            state_current = np.array([current_location/self.sates_len, self.state_record[current_location - self.nx]/self.action_num, 1])
        else:
            state_current = np.array([current_location/self.sates_len, self.state_record[current_location - self.nx]/self.action_num, action/self.action_num])
            
        return state_current, reward, action, mask_new

    def step_all(self, action, current_location,csv_design,csv_fab):

        x_loc = current_location % self.nx
        y_loc = current_location // self.nx
        # action = mapping_types(current_location, self.room_types, self.state_record, self.nx, action)

        if action == self.ind_stair[0] and self.stair_used == 0:
            self.stair_used = 1
            self.delet_ind.extend(self.ind_stair)
            self.coverd_stair = [current_location + self.nx - 1, current_location + self.nx * 2 - 1]
        if action == self.ind_stair[1] and self.stair_used == 0:
            self.stair_used = 1
            self.delet_ind.extend(self.ind_stair)
            self.coverd_stair = [current_location + self.nx - 1, current_location - 1]
            
        self.state_record[current_location - 1] = int(action)

        self.eval_design.count_type(action)
        self.eval_fab.count_type(action)

        self.reward_Design = self.eval_design.compute_design_reward(self.stair_used,csv_design)
        self.reward_Fabrication, self.fab_time = self.eval_fab.compute_fabrication_reward(self.stair_used,csv_fab)

        if y_loc == 0:
            state_current = np.array([current_location/self.sates_len, 1, action/self.action_num])
        elif x_loc == 0:
            state_current = np.array([current_location/self.sates_len, self.state_record[current_location - self.nx]/self.action_num, 1])
        else:
            state_current = np.array([current_location/self.sates_len, self.state_record[current_location - self.nx]/self.action_num, action/self.action_num])
            
        return state_current, self.reward_Design + self.reward_Fabrication, action
    
    def step_whole_canvas(self, action, current_location,csv_fab):

        x_loc = current_location % self.nx
        y_loc = current_location // self.nx
        action = mapping_types(current_location, self.room_types, self.state_record, self.nx, action)

        if action == self.ind_stair[0]:
            self.stair_used = 1
            self.delet_ind.extend(self.ind_stair)
            self.coverd_stair = [current_location + self.nx - 1, current_location + self.nx * 2 - 1]
        if action == self.ind_stair[1]:
            self.stair_used = 1
            self.delet_ind.extend(self.ind_stair)
            self.coverd_stair = [current_location + self.nx - 1, current_location - 1]
            
        self.state_record[current_location - 1] = action

        self.eval_design.count_type(action)
        self.eval_fab.count_type(action)

        reward = self.eval_fab.compute_single_fabrication_reward(action,csv_fab)

        state_current = self.state_record/self.action_num
          
        return state_current, reward, action

    def step_whole_canvas_all(self, action, current_location,csv_design,csv_fab):

        action = mapping_types(current_location, self.room_types, self.state_record, self.nx, action)

        if action == self.ind_stair[0]:
            self.stair_used = 1
            self.delet_ind.extend(self.ind_stair)
            self.coverd_stair = [current_location + self.nx - 1, current_location + self.nx * 2 - 1]
        if action == self.ind_stair[1]:
            self.stair_used = 1
            self.delet_ind.extend(self.ind_stair)
            self.coverd_stair = [current_location + self.nx - 1, current_location - 1]
            
        self.state_record[current_location - 1] = action

        self.eval_design.count_type(action)
        self.eval_fab.count_type(action)

        self.reward_Design = self.eval_design.compute_design_reward(self.stair_used,csv_design)
        self.reward_Fabrication, self.fab_time  = self.eval_fab.compute_fabrication_reward(self.stair_used,csv_fab)

        state_current = self.state_record/self.action_num
            
        return state_current, self.reward_Design + self.reward_Fabrication, action
