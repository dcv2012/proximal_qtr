import numpy as np
import pandas as pd
import torch

origin_para_set = {
    'mu_Y0': -0.35,
    'sigma_Y0': 0.2,
    'mu_U0': 0.35,
    'sigma_U0': 0.5,
    
    # Propensity near 0.5 but mildly confounded to preserve PIPW policy value
    'alpha_A1': [-0.2, 0.05, 0.5],
    
    # Z1 is a STRONG proxy for U0
    'mu_Z11': [0.2, 0.5, 0.5, 2.5],
    'sigma_Z11': 0.2,
    'mu_Z12': [-0.1, 0.3, 0.45, -2.5],
    'sigma_Z12': 0.2,
    
    # W1 is a WEAK proxy for U0 (small coeff, high noise) -> Hurts h21 (POR)
    'mu_W11': [0.2, -0.35, 0.8],
    'sigma_W11': 0.5, #origin 1.8
    
    # Y_1, U_1
    'mu_Y1': [0.2, 0.2, 0.4, -0.6],
    'sigma_Y1': 0.5,
    'mu_U1': [0.1, 0.2, 0, 0.8],
    'sigma_U1': 0.5,
    
    # A2 propensity mildly confounded by U1, completely saving PIPW from extreme weights
    'alpha_A2': [-0.2, 0.0, 0.02, 0.03, -0.03, 0.5],
    
    # Z2 is a STRONG proxy for U1 -> Helps q22 (PIPW)
    'mu_Z21': [0.2, 0, 0, 0.2, 0.2, 2.5, 0.2, 0.2, -0.5],
    'sigma_Z21': 1.0,
    'mu_Z22': [-0.1, 0.1, 0.1, 0.2, -0.2, -0.5, -0.1, 0.2, 0.5],
    'sigma_Z22': 1.0,
    
    # W2 is a STRONG proxy for U1 -> Helps h22 (PHA, PMR, cPMR)
    'mu_W21': [0.35, 0.2, 0.5, 2.5, -0.2, 0.2],
    'sigma_W21': 0.2,
    'mu_W22': [0.2, 0.1, 0.5, -2.5, -0.5, -0.2],
    'sigma_W22': 0.2,
    
    # Huge U effect on Y2 to destroy Naive Estimation. Treat A=1 is actually terrible.
    # ['ones', 'A2', 'A1', 'A2*A1', 'W21', 'W22', 'W11', 'Y1', 'U1', 'Y0', 'U0', 'A1*U0', 'A2*U1', 'A1*Y0', 'A2*Y1']
    'mu_Y2': [1.0, -1.5, -2.0, -2.0, 0, 0, 0, -1.5, 0, 0, 2, 0, 5, 0, 3],
    'sigma_Y2': 0.2

    ## optimal q*: 0.25-1.2112, 0.5-5.38
}


def adjust_para_set_for_new_coding(original_para: dict) -> dict:
    """
    将 para_set 中的参数进行调整，以适应 A 从 {0,1} 变为 {-1,1} 的编码，
    同时保持下游变量的统计特性不变。
    """
    new_para = original_para.copy()
    
    def update_coeffs(arr, idx_A):
        new_arr = np.array(arr, dtype=float).copy()
        beta_A = new_arr[idx_A]
        new_arr[0] = new_arr[0] + 0.5 * beta_A
        new_arr[idx_A] = 0.5 * beta_A
        return new_arr

    # A1 is index 1 for Z11, Z12, Y1, U1, A2
    new_para['mu_Z11'] = update_coeffs(original_para['mu_Z11'], 1)
    new_para['mu_Z12'] = update_coeffs(original_para['mu_Z12'], 1)
    new_para['mu_Y1']  = update_coeffs(original_para['mu_Y1'], 1)
    new_para['mu_U1']  = update_coeffs(original_para['mu_U1'], 1)
    new_para['alpha_A2'] = update_coeffs(original_para['alpha_A2'], 1) 
    
    # Z2 requires update on A2 (idx 3) and A1 (idx 6)
    def update_z2_coeffs(arr):
        new_arr = np.array(arr, dtype=float).copy()
        beta_A2 = new_arr[3]
        beta_A1 = new_arr[6]
        new_arr[0] = new_arr[0] + 0.5*beta_A2 + 0.5*beta_A1
        new_arr[3] = 0.5*beta_A2
        new_arr[6] = 0.5*beta_A1
        return new_arr
        
    new_para['mu_Z21'] = update_z2_coeffs(original_para['mu_Z21'])
    new_para['mu_Z22'] = update_z2_coeffs(original_para['mu_Z22'])
    
    def update_y2_coeffs(arr):
        new_arr = np.array(arr, dtype=float).copy()
        beta_A2 = new_arr[1]
        beta_A1 = new_arr[2]
        beta_Int = new_arr[3]
        
        new_arr[0] = new_arr[0] + 0.5*beta_A2 + 0.5*beta_A1 + 0.25*beta_Int
        new_arr[1] = 0.5*beta_A2 + 0.25*beta_Int
        new_arr[2] = 0.5*beta_A1 + 0.25*beta_Int
        new_arr[3] = 0.25*beta_Int
        
        if len(new_arr) > 11:
            beta_A1U0 = new_arr[11]
            beta_A2U1 = new_arr[12]
            beta_A1Y0 = new_arr[13]
            beta_A2Y1 = new_arr[14]
            
            new_arr[11] = 0.5 * beta_A1U0
            new_arr[12] = 0.5 * beta_A2U1
            new_arr[13] = 0.5 * beta_A1Y0
            new_arr[14] = 0.5 * beta_A2Y1
            
            new_arr[10] += 0.5 * beta_A1U0  # U0
            new_arr[8]  += 0.5 * beta_A2U1  # U1
            new_arr[9]  += 0.5 * beta_A1Y0  # Y0
            new_arr[7]  += 0.5 * beta_A2Y1  # Y1
            
        return new_arr
        
    new_para['mu_Y2'] = update_y2_coeffs(original_para['mu_Y2'])
    
    return new_para


def data_gen(sample_size: int, para_set: dict) -> pd.DataFrame:
    """生成观测数据集"""
    N = sample_size
    
    Y0 = np.random.normal(para_set['mu_Y0'], para_set['sigma_Y0'], N)
    U0 = np.random.normal(para_set['mu_U0'], para_set['sigma_U0'], N)
    
    # A1
    design_A1 = np.column_stack([np.ones(N), Y0, U0])
    lin_pred_A1 = design_A1 @ np.array(para_set['alpha_A1'])
    prop_score_1 = 1 / (1 + np.exp(-lin_pred_A1))
    A1_bin = np.random.binomial(1, prop_score_1, N)
    A1 = 2 * A1_bin - 1
    
    # Z11, Z12
    design_Z1 = np.column_stack([np.ones(N), A1, Y0, U0])
    Z11 = design_Z1 @ np.array(para_set['mu_Z11']) + np.random.normal(0, para_set['sigma_Z11'], N)
    Z12 = design_Z1 @ np.array(para_set['mu_Z12']) + np.random.normal(0, para_set['sigma_Z12'], N)
    
    # W11
    design_W1 = np.column_stack([np.ones(N), Y0, U0])
    W11 = design_W1 @ np.array(para_set['mu_W11']) + np.random.normal(0, para_set['sigma_W11'], N)
    
    # Y1, U1
    design_Y1 = np.column_stack([np.ones(N), A1, Y0, U0])
    Y1 = design_Y1 @ np.array(para_set['mu_Y1']) + np.random.normal(0, para_set['sigma_Y1'], N)
    U1 = design_Y1 @ np.array(para_set['mu_U1']) + np.random.normal(0, para_set['sigma_U1'], N)
    
    # A2
    design_A2 = np.column_stack([np.ones(N), A1, Y0, U0, Y1, U1])
    lin_pred_A2 = design_A2 @ np.array(para_set['alpha_A2'])
    prop_score_2 = 1 / (1 + np.exp(-lin_pred_A2))
    A2_bin = np.random.binomial(1, prop_score_2, N)
    A2 = 2 * A2_bin - 1
    
    # Z21, Z22
    design_Z2 = np.column_stack([np.ones(N), Z11, Z12, A2, Y1, U1, A1, Y0, U0])
    Z21 = design_Z2 @ np.array(para_set['mu_Z21']) + np.random.normal(0, para_set['sigma_Z21'], N)
    Z22 = design_Z2 @ np.array(para_set['mu_Z22']) + np.random.normal(0, para_set['sigma_Z22'], N)
    
    # W21, W22
    design_W2 = np.column_stack([np.ones(N), W11, Y1, U1, Y0, U0])
    W21 = design_W2 @ np.array(para_set['mu_W21']) + np.random.normal(0, para_set['sigma_W21'], N)
    W22 = design_W2 @ np.array(para_set['mu_W22']) + np.random.normal(0, para_set['sigma_W22'], N)
    
    # Y2
    design_Y2 = np.column_stack([np.ones(N), A2, A1, A2 * A1, W21, W22, W11, Y1, U1, Y0, U0, A1*U0, A2*U1, A1*Y0, A2*Y1])
    Y2 = design_Y2 @ np.array(para_set['mu_Y2']) + np.random.normal(0, para_set['sigma_Y2'], N)
    
    df = pd.DataFrame({
        'Y0': Y0, 'U0': U0, 'A1': A1, 
        'Z11': Z11, 'Z12': Z12, 'W11': W11,
        'Y1': Y1, 'U1': U1, 'A2': A2, 
        'Z21': Z21, 'Z22': Z22, 'W21': W21, 'W22': W22, 'Y2': Y2
    })
    return df


def intervened_data_gen(sample_size: int, para_set: dict, a: list = [1, 1]) -> pd.DataFrame:
    """干预反事实数据生成 (固定A序列)"""
    N = sample_size
    Y0 = np.random.normal(para_set['mu_Y0'], para_set['sigma_Y0'], N)
    U0 = np.random.normal(para_set['mu_U0'], para_set['sigma_U0'], N)
    
    A1 = np.full(N, a[0])
    
    design_Z1 = np.column_stack([np.ones(N), A1, Y0, U0])
    Z11 = design_Z1 @ np.array(para_set['mu_Z11']) + np.random.normal(0, para_set['sigma_Z11'], N)
    Z12 = design_Z1 @ np.array(para_set['mu_Z12']) + np.random.normal(0, para_set['sigma_Z12'], N)
    
    design_W1 = np.column_stack([np.ones(N), Y0, U0])
    W11 = design_W1 @ np.array(para_set['mu_W11']) + np.random.normal(0, para_set['sigma_W11'], N)
    
    design_Y1 = np.column_stack([np.ones(N), A1, Y0, U0])
    Y1 = design_Y1 @ np.array(para_set['mu_Y1']) + np.random.normal(0, para_set['sigma_Y1'], N)
    U1 = design_Y1 @ np.array(para_set['mu_U1']) + np.random.normal(0, para_set['sigma_U1'], N)
    
    A2 = np.full(N, a[1])
    
    design_Z2 = np.column_stack([np.ones(N), Z11, Z12, A2, Y1, U1, A1, Y0, U0])
    Z21 = design_Z2 @ np.array(para_set['mu_Z21']) + np.random.normal(0, para_set['sigma_Z21'], N)
    Z22 = design_Z2 @ np.array(para_set['mu_Z22']) + np.random.normal(0, para_set['sigma_Z22'], N)
    
    design_W2 = np.column_stack([np.ones(N), W11, Y1, U1, Y0, U0])
    W21 = design_W2 @ np.array(para_set['mu_W21']) + np.random.normal(0, para_set['sigma_W21'], N)
    W22 = design_W2 @ np.array(para_set['mu_W22']) + np.random.normal(0, para_set['sigma_W22'], N)
    
    design_Y2 = np.column_stack([np.ones(N), A2, A1, A2 * A1, W21, W22, W11, Y1, U1, Y0, U0, A1*U0, A2*U1, A1*Y0, A2*Y1])
    Y2 = design_Y2 @ np.array(para_set['mu_Y2']) + np.random.normal(0, para_set['sigma_Y2'], N)
    
    df = pd.DataFrame({
        'Y0': Y0, 'U0': U0, 'A1': A1, 
        'Z11': Z11, 'Z12': Z12, 'W11': W11,
        'Y1': Y1, 'U1': U1, 'A2': A2, 
        'Z21': Z21, 'Z22': Z22, 'W21': W21, 'W22': W22, 'Y2': Y2
    })
    return df


def dynamic_intervened_data_gen(sample_size: int, para_set: dict, f1=None, f2=None, device='cpu', seed=None) -> pd.DataFrame:
    """动态策略控制的反事实数据生成"""
    if seed is not None:
        np.random.seed(seed)
    N = sample_size
    Y0 = np.random.normal(para_set['mu_Y0'], para_set['sigma_Y0'], N)
    U0 = np.random.normal(para_set['mu_U0'], para_set['sigma_U0'], N)
    
    with torch.no_grad():
        if f1 is not None:
            H1_input = torch.tensor(Y0, dtype=torch.float32).unsqueeze(1).to(device)
            A1_pred = np.sign(f1(H1_input).cpu().numpy().flatten())
            A1_pred[A1_pred == 0] = 1
            A1 = A1_pred
        else:
            A1 = np.ones(N)
            
    design_Z1 = np.column_stack([np.ones(N), A1, Y0, U0])
    Z11 = design_Z1 @ np.array(para_set['mu_Z11']) + np.random.normal(0, para_set['sigma_Z11'], N)
    Z12 = design_Z1 @ np.array(para_set['mu_Z12']) + np.random.normal(0, para_set['sigma_Z12'], N)
    
    design_W1 = np.column_stack([np.ones(N), Y0, U0])
    W11 = design_W1 @ np.array(para_set['mu_W11']) + np.random.normal(0, para_set['sigma_W11'], N)
    
    design_Y1 = np.column_stack([np.ones(N), A1, Y0, U0])
    Y1 = design_Y1 @ np.array(para_set['mu_Y1']) + np.random.normal(0, para_set['sigma_Y1'], N)
    U1 = design_Y1 @ np.array(para_set['mu_U1']) + np.random.normal(0, para_set['sigma_U1'], N)
    
    with torch.no_grad():
        if f2 is not None:
            H2_input = torch.cat([
                torch.tensor(Y0, dtype=torch.float32).unsqueeze(1),
                torch.tensor(Y1, dtype=torch.float32).unsqueeze(1),
                torch.tensor(A1, dtype=torch.float32).unsqueeze(1)
            ], dim=1).to(device)
            A2_pred = np.sign(f2(H2_input).cpu().numpy().flatten())
            A2_pred[A2_pred == 0] = 1
            A2 = A2_pred
        else:
            A2 = np.ones(N)
    
    design_Z2 = np.column_stack([np.ones(N), Z11, Z12, A2, Y1, U1, A1, Y0, U0])
    Z21 = design_Z2 @ np.array(para_set['mu_Z21']) + np.random.normal(0, para_set['sigma_Z21'], N)
    Z22 = design_Z2 @ np.array(para_set['mu_Z22']) + np.random.normal(0, para_set['sigma_Z22'], N)
    
    design_W2 = np.column_stack([np.ones(N), W11, Y1, U1, Y0, U0])
    W21 = design_W2 @ np.array(para_set['mu_W21']) + np.random.normal(0, para_set['sigma_W21'], N)
    W22 = design_W2 @ np.array(para_set['mu_W22']) + np.random.normal(0, para_set['sigma_W22'], N)
    
    design_Y2 = np.column_stack([np.ones(N), A2, A1, A2 * A1, W21, W22, W11, Y1, U1, Y0, U0, A1*U0, A2*U1, A1*Y0, A2*Y1])
    Y2 = design_Y2 @ np.array(para_set['mu_Y2']) + np.random.normal(0, para_set['sigma_Y2'], N)
    
    df = pd.DataFrame({
        'Y0': Y0, 'U0': U0, 'A1': A1, 
        'Z11': Z11, 'Z12': Z12, 'W11': W11,
        'Y1': Y1, 'U1': U1, 'A2': A2, 
        'Z21': Z21, 'Z22': Z22, 'W21': W21, 'W22': W22, 'Y2': Y2
    })
    return df