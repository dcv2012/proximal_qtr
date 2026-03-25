import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal, uniform, randint

from src.data.data_class import MMRTrainDataSet_q, MMRTestDataSet_q, MMRTrainDataSet_q_2stage, MMRTestDataSet_q_2stage
from src.data.data_generate import origin_para_set, adjust_para_set_for_new_coding, data_gen, intervened_data_gen

from src.utils.setseed import set_seed

set_seed(20048)

# get new parameter set for A in {-1,1}
new_para_set = adjust_para_set_for_new_coding(origin_para_set)

def generate_data(n_sample: int, scenario: str = 'S1'):
    data = data_gen(n_sample, new_para_set)
    Y0_arr = np.array(data['Y0'])   # X
    A1_arr = np.array(data['A1'])
    W1_arr = np.array(data['W1'])
    Z1_arr= np.array(data['Z1'])
    Y1_arr = np.array(data['Y1'])
    
    A2_arr = np.array(data['A2'])
    W2_arr = np.array(data['W2'])
    Z2_arr= np.array(data['Z2'])
    Y2_arr = np.array(data['Y2'])
    
    intervened_data_1p = intervened_data_gen(n_sample, new_para_set, [1,1])
    intervened_data_1n = intervened_data_gen(n_sample, new_para_set, [-1,1])
    Y1p_arr_= np.array(intervened_data_1p['Y1'])
    Y1n_arr_= np.array(intervened_data_1n['Y1'])
    Ep = Y1p_arr_.mean()
    En = Y1n_arr_.mean()
    
    return {
            'A1': A1_arr.reshape(-1, 1),
            'W1': W1_arr.reshape(-1, 1),
            'Y0': Y0_arr.reshape(-1, 1),
            'Y1': Y1_arr.reshape(-1, 1),
            'Z1': Z1_arr.reshape(-1, 1),
            'A2': A2_arr.reshape(-1, 1),
            'W2': W2_arr.reshape(-1, 1),
            'Y2': Y2_arr.reshape(-1, 1),
            'Z2': Z2_arr.reshape(-1, 1),
            'E11': Ep,
            'E10': En
    }
    


# Generate training data
def generate_train_simulation_q(n_sample: int, scenario: str = 'S1', **kwargs):
    data = generate_data(n_sample, scenario=scenario)
    A1 = data['A1']
    W1 = data['W1']
    Y0 = data['Y0']
    Y1 = data['Y1']
    Z1 = data['Z1'] 
    A2 = data['A2']
    W2 = data['W2']
    Y2 = data['Y2']
    Z2 = data['Z2']
    
    A1_target = np.zeros(n_sample) # all set to 0
    return MMRTrainDataSet_q(treatment=A1,
                             treatment_target=A1_target,
                             treatment_proxy=Z1,
                             outcome_proxy=W1,
                             outcome=Y1,
                             backdoor=Y0)

# Generate test data
def generate_test_simulation_q(n_sample: int, scenario: str, **kwargs):
    data = generate_data(n_sample, scenario=scenario)
    A1 = data['A1']
    W1 = data['W1']
    Y0 = data['Y0']
    Y1 = data['Y1']
    Z1 = data['Z1']
    A2 = data['A2']
    W2 = data['W2']
    Y2 = data['Y2']
    Z2 = data['Z2']
    E11 = data['E11']
    E10 = data['E10']
    return MMRTestDataSet_q(treatment=A1,
                           treatment_proxy=Z1,
                           outcome_proxy=W1,
                           outcome=Y1,
                           backdoor=Y0,
                           structural=[E11, E10])

def generate_train_simulation_q_2stage(n_sample: int, scenario: str = 'S1', **kwargs):
    data = generate_data(n_sample, scenario=scenario)
    A1 = data['A1']
    W1 = data['W1']
    Y0 = data['Y0']
    Y1 = data['Y1']
    Z1 = data['Z1']
    A2 = data['A2']
    W2 = data['W2']
    Y2 = data['Y2']
    Z2 = data['Z2']

    A1_target = np.zeros(n_sample) # all set to -1
    A2_target = np.zeros(n_sample) # all set to -1
    
    return MMRTrainDataSet_q_2stage(
        treatment1=A1,
        treatment_target1=A1_target,
        treatment_proxy1=Z1,
        outcome_proxy1=W1,
        outcome1=Y1,
        backdoor1=Y0,
        treatment2=A2,
        treatment_target2=A2_target,
        treatment_proxy2=Z2,
        outcome_proxy2=W2,
        outcome2=Y2,
        backdoor2=Y1
    )

def generate_test_simulation_q_2stage(n_sample: int, scenario: str, **kwargs):
    data = generate_data(n_sample, scenario=scenario)
    A1 = data['A1']
    W1 = data['W1']
    Y0 = data['Y0']
    Y1 = data['Y1']
    Z1 = data['Z1']
    A2 = data['A2']
    W2 = data['W2']
    Y2 = data['Y2']
    Z2 = data['Z2']
    E11 = data['E11']
    E10 = data['E10']
    
    return MMRTestDataSet_q_2stage(
        treatment1=A1,
        treatment_proxy1=Z1,
        outcome_proxy1=W1,
        outcome1=Y1,
        backdoor1=Y0,
        structural1=[E11, E10],
        treatment2=A2,
        treatment_proxy2=Z2,
        outcome_proxy2=W2,
        outcome2=Y2,
        backdoor2=Y1
    )
    


def get_data_from_mmr(mmr_data) -> dict[str, np.ndarray]:
    # 根据输入的数据类型（mmrtrain/test），返回一个包含两阶段数据的字典
    return {
            'A1': mmr_data.treatment1,
            'W1': mmr_data.treatment_proxy1,
            'Y0': mmr_data.backdoor1,
            'Y1': mmr_data.outcome1,
            'Z1': mmr_data.outcome_proxy1,
            'A2': mmr_data.treatment2,
            'W2': mmr_data.treatment_proxy2,
            'Y2': mmr_data.outcome2,
            'Z2': mmr_data.outcome_proxy2,
    }
    
    
    

if __name__ == "__main__":
    set_seed(20048)
    
    mtd = generate_train_simulation_q_2stage(1000, scenario='S1')
    data = get_data_from_mmr(mtd)
    
    s = mtd.treatment1 - data['A1']
    print(s)
    