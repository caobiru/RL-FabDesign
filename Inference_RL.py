import os
import sys
import numpy as np
import pandas as pd
import os
import Team_Project.Agent_Project as Agent_Model
import Team_Project.ENV_Project as ENV_Model
from RL_Base.DQN_Base import plotLearning
from RL_Base.ENV_Base import parse_elsx
import warnings
warnings.filterwarnings("ignore")
sys.dont_write_bytecode = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    main_path = os.getcwd()
    ddqn_scores = []
    eps_history = []
    ddqn_ave_scores = []

    # When Design Libs changed, adjust the following paras
    n_actions = 16
    Num_roomTypes = 5
    ind_stair = [14, 15]

    # canvas
    Num_x = 6
    Num_y = 8
    all_canvas = int(Num_x * Num_y)

    # RL paras
    n_games = 10
    store_actions = 10
    store_gap = int(n_games/store_actions)
    input_dims = 3# set to = all_canvas if step_whole_canvas
    batch_size = 128

    # rules
    adjacent_rules = parse_elsx(pd.read_excel(os.path.join(main_path,'Team_Project','adjacent_rules.xlsx')).values)
    fabrication_time_info = pd.read_csv(os.path.join(main_path,'Team_Project','fabrication_time_info.csv')).values
    fab_type = pd.read_excel(os.path.join(main_path,'Team_Project','fabrication_type_list.xlsx')).values

    # goals
    room_area_goal = pd.read_csv(os.path.join(main_path,'Team_Project','room_area_goal.csv')).values
    time_goal = 30

    # init
    Agent = Agent_Model.Agent(n_actions,batch_size,input_dims)
    Agent.update_parameters()
    Agent.load_model('saved_model_10000')
    ENV = ENV_Model.EVN(n_actions,input_dims,Num_roomTypes,Num_x,Num_y,adjacent_rules,ind_stair,
                        fab_type[0],fab_type[1],fab_type[2],time_goal)

    action_list = np.zeros((store_actions, all_canvas), dtype=np.int32)

    for i in range(n_games):

        observation = ENV.reset_project()
        score = 0

        for j in range(all_canvas):
            mask = ENV.get_rule_to_mask(j)
            action = Agent.inference_action(observation, mask)

            if j < all_canvas - 1:
                observation_, reward, action, _ = ENV.step_fab_only(action, int(j + 1),fabrication_time_info)
                score += reward
                observation = observation_
            else:  # last step
                _, reward, action = ENV.step_all(action, int(j + 1),room_area_goal,fabrication_time_info)
                score += reward

            if i % store_gap == 0:
                action_list[i//store_gap, j] = action

        eps_history.append(Agent.epsilon)
        ddqn_scores.append(score)
        avg_score = np.mean(ddqn_scores[-100:])
        ddqn_ave_scores.append(avg_score)

        print('episode: ', i, 'Des_R: %d' % ENV.reward_Design,'Fab_R: %d' % ENV.reward_Fabrication,'Fab_T: %d' % ENV.fab_time,'Score: %.2f' % score,
            'epsilon: %.2f' % Agent.epsilon,'Ave_Score %.2f' % avg_score)
        
    filename = 'ddqn_trained_local_%d_each.png' % (n_games)
    filename2 = 'ddqn_trained_local_%d_ave.png' % (n_games)
    x = [i for i in range(n_games)]
    plotLearning(x, ddqn_scores, eps_history, filename)
    plotLearning(x, ddqn_ave_scores, eps_history, filename2)

    outcome = pd.DataFrame(np.array(action_list).reshape(store_actions,Num_x*Num_y))
    outcome.to_csv('trained_output_%d.csv' % (n_games))

if __name__ == '__main__':
    main()
