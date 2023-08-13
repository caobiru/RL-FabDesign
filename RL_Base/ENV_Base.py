import numpy as np
from abc import ABCMeta
import Design_Evaluation.Design_Evaluation_Base as DesignReward_Function
import Fabrication_Evaluation.Fabrication_Evaluation_Base as FabricationReward_Function

def parse_elsx(df):
    for m,n in enumerate(df):
        for i,j in enumerate(n):
            if not isinstance(j,int):
                matched = []
                for f in j:
                    try: f = int(f)
                    except: pass
                    if isinstance(f,int):
                        matched.append(f)
                df[m,i] = matched
    return df

class EVN_Base(metaclass=ABCMeta):

    def __init__(self, n_x=6, n_y=8):
        self.nx = n_x
        self.ny = n_y
        self.state_record = np.zeros(int(self.nx * self.ny))
        self.all_state = int(self.nx * self.ny)
        self.reward_Design = 0
        self.reward_Fabrication = 0
        self.fab_time = 0



        




