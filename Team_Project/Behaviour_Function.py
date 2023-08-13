import numpy as np

def checkleft(left, rule, delet_ind):
    left = int(left)
    match_id = rule[left, 1]

    matched = []
    for i in range(rule.shape[0]):
        if i not in delet_ind:
            attach = rule[i, 3]
            if np.intersect1d(np.array(match_id), np.array([attach])).tolist() != []:
                matched.append(i)
    return np.array(matched)


def checkdown(down, rule, delet_ind):
    down = int(down)
    match_id = rule[down, 2]
    matched = []
    for i in range(rule.shape[0]):
        if i not in delet_ind:
            attach = rule[i, 0]
            if np.intersect1d(np.array(match_id), np.array([attach])).tolist() != []:
                matched.append(i)
    return np.array(matched)


def checkleftdown(left, down, rule, delet_ind):
    left = int(left)
    down = int(down)
    match_id_left = rule[left, 1]
    match_id_down = rule[down, 2]
    matched_left = []
    matched_down = []
    for i in range(rule.shape[0]):
        if i not in delet_ind:
            attach_left = rule[i, 3]
            attach_down = rule[i, 0]
            if np.intersect1d(np.array(match_id_left), np.array([attach_left])).tolist() != []:
                matched_left.append(i)
            if np.intersect1d(np.array(match_id_down), np.array([attach_down])).tolist() != []:
                matched_down.append(i)
    matched_final = list(set(matched_left).intersection(set(matched_down)))
    return np.array(matched_final)

def mask_actions(actions,actions_space,action_num,mask,batch):
    max_act = np.zeros(batch, dtype=np.int32)
    for idx, act in enumerate(actions):
        mask_onehot = mask[idx]
        mask_out = np.argwhere(mask_onehot==0)
        # mask_out = np.setdiff1d(actions_space, mask_list, assume_unique=False)
        actions_masked = act.reshape(action_num)
        actions_masked[mask_out] = -999
        max_act[idx] = np.argmax(actions_masked)
    return max_act

def mapping_types(current, num_type, all_state, nx, action):
    current = current - 1
    x = current % nx
    y = current // nx
    all_types = [i for i in range(num_type)]
    if x == 0 and y != 0:
        if all_state[current - nx] in all_types and action in all_types:
            action = all_state[current - nx]
    elif x != 0:
        if all_state[current - 1] in all_types and action in all_types:
            action = all_state[current - 1]
        elif y != 0 and all_state[current - nx] in all_types and action in all_types:
            action = all_state[current - nx]

    return int(action)

def check_distance(current_num, num_range):
    if num_range[0] <= current_num <= num_range[1]:
        return 30
    elif current_num > num_range[1]:
        return (num_range[1] - current_num)*5
    else:
        return (current_num - num_range[0])*10

def map_reward(data,range):
    a = -1
    b = 1
    min,max = range

    return a + (b-a)/(max-min)*(data-min)
