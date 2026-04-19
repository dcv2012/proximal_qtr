import numpy as np
import pandas as pd
import torch
from scipy.stats import norm, multivariate_normal, uniform, randint
import scipy.stats as stats

# 参数设置 (Parameter Setting)
origin_para_set = {
    'mu_Y0': -0.35, 'sigma_Y0': 0.5,
    'mu_U0': 0.35, 'sigma_U0': 0.5,
    
    # A1 的系数: intercept, Y0, U0
    'alpha_A1': np.array([-0.5, -0.35, 0.35]),
    
    # Z1 的系数: intercept, A1, Y0, U0
    'mu_Z1': np.array([0.2, 0.5, 0.5, 0.75]), 'sigma_Z1': 0.5,
    
    # W1 的系数: intercept, Y0, U0
    'mu_W1': np.array([0.2, 0.5, -0.95]), 'sigma_W1': 0.5,
    
    # Y1 的系数: intercept, A1, Y0, U0
    'mu_Y1': np.array([0.2, 0.7, 0.7, 0]), 'sigma_Y1': 0.5,
    
    # U1 的系数: intercept, A1, Y0, U0
    'mu_U1': np.array([0.2, 0.7, 0, 0.7]), 'sigma_U1': 0.5,
    
    # A2 的系数: intercept, A1, Y0, U0, Y1, U1
    # 注意：R代码中 cbind(1, A1, Y0, U0, Y1, U1) 的顺序
    'alpha_A2': np.array([-0.5, 0.5, -0.35, 0.35, -0.35, 0.35]),
    
    # Z2 的系数: intercept, Z1, A2, Y1, U1, A1, Y0, U0
    'mu_Z2': np.array([0.2, 0, 0.5, 0.5, -0.75, 0.5, 0.5, -0.75]), 'sigma_Z2': 0.5,
    
    # W2 的系数: intercept, W1, Y1, U1, Y0, U0
    'mu_W2': np.array([0.35, 0, 0.45, -0.85, 0.45, -0.85]), 'sigma_W2': 0.5,
    
    # Y2 的系数: intercept, A2, A1, A2*A1, W2, W1, Y1, U1, Y0, U0
    'mu_Y2': np.array([-1.3, 1, 1.14, 0, 0, 0, 0.5, -0.7, 0.2, -0.7]), 'sigma_Y2': 0.5
}



# Parameter adjustment function for new coding of A
def adjust_para_set_for_new_coding(original_para: dict) -> dict:
    """
    将 para_set 中的参数进行调整，以适应 A 从 {0,1} 变为 {-1,1} 的编码，
    同时保持下游变量的统计特性不变。
    """
    # 深拷贝以避免修改原字典
    new_para = original_para.copy()
    
    # 辅助函数：更新单变量依赖的系数数组
    # arr: 系数数组
    # idx_A: A 在设计矩阵中的索引位置（从0开始）
    def update_coeffs(arr, idx_A):
        new_arr = arr.copy()
        beta_A = arr[idx_A]
        # 更新截距 (总是位于索引0)
        new_arr[0] = arr[0] + 0.5 * beta_A
        # 更新 A 的系数
        new_arr[idx_A] = 0.5 * beta_A
        return new_arr

    # 1. mu_Z1: design matrix [1, A1, Y0, U0]
    # A1 index = 1
    new_para['mu_Z1'] = update_coeffs(original_para['mu_Z1'], 1)
    
    # 2. mu_Y1: design matrix [1, A1, Y0, U0]
    # A1 index = 1
    new_para['mu_Y1'] = update_coeffs(original_para['mu_Y1'], 1)
    
    # 3. mu_U1: design matrix [1, A1, Y0, U0]
    # A1 index = 1
    new_para['mu_U1'] = update_coeffs(original_para['mu_U1'], 1)
    
    # 4. alpha_A2: design matrix [1, A1, Y0, U0, Y1, U1]
    # A1 index = 1
    # 注意：这里调整是为了让 A2=1 的倾向性评分(概率)保持不变
    new_para['alpha_A2'] = update_coeffs(original_para['alpha_A2'], 1)
    
    # 5. mu_Z2: design matrix [1, Z1, A2, Y1, U1, A1, Y0, U0]
    # 涉及 A2 (index 2) 和 A1 (index 5)
    mu_Z2 = original_para['mu_Z2'].copy()
    beta_A2 = mu_Z2[2]
    beta_A1 = mu_Z2[5]
    
    # 截距更新：加上 A2 和 A1 的贡献
    mu_Z2[0] = mu_Z2[0] + 0.5 * beta_A2 + 0.5 * beta_A1
    # 系数更新
    mu_Z2[2] = 0.5 * beta_A2
    mu_Z2[5] = 0.5 * beta_A1
    new_para['mu_Z2'] = mu_Z2
    
    # 6. mu_Y2: design matrix [1, A2, A1, A2*A1, W2, W1, Y1, U1, Y0, U0]
    # Indices: Int=0, A2=1, A1=2, Interaction=3
    mu_Y2 = original_para['mu_Y2'].copy()
    beta_A2 = mu_Y2[1]
    beta_A1 = mu_Y2[2]
    beta_Int = mu_Y2[3] # 原设定中为0
    
    # 公式转换
    # New Intercept
    mu_Y2[0] = mu_Y2[0] + 0.5*beta_A2 + 0.5*beta_A1 + 0.25*beta_Int
    # New A2
    mu_Y2[1] = 0.5*beta_A2 + 0.25*beta_Int
    # New A1
    mu_Y2[2] = 0.5*beta_A1 + 0.25*beta_Int
    # New Interaction
    mu_Y2[3] = 0.25*beta_Int
    
    new_para['mu_Y2'] = mu_Y2
    
    return new_para



# 观测数据生成函数 (Observed Data Generation)
def data_gen(sample_size: int, para_set: dict) -> pd.DataFrame:
    N = sample_size
    
    # 生成 Y0, U0
    Y0 = np.random.normal(para_set['mu_Y0'], para_set['sigma_Y0'], N)
    U0 = np.random.normal(para_set['mu_U0'], para_set['sigma_U0'], N)
    
    # 生成 A1 (基于 propensity score 的二项分布)
    # design matrix: [1, Y0, U0]
    design_A1 = np.column_stack((np.ones(N), Y0, U0))
    prop_score_1 = 1 / (1+np.exp(np.dot(design_A1, para_set['alpha_A1'])))
    A1_bin = np.random.binomial(1, prop_score_1, N)
    # map A1 from {0,1} to {-1,1}
    A1 = 2 * A1_bin - 1 
    # A1 = A1_bin
    
    # 生成 Z1
    # design matrix: [1, A1, Y0, U0]
    design_Z1 = np.column_stack((np.ones(N), A1, Y0, U0))
    Z1 = np.dot(design_Z1, para_set['mu_Z1']) + np.random.normal(0, para_set['sigma_Z1'], N)
    
    # 生成 W1
    # design matrix: [1, Y0, U0]
    design_W1 = np.column_stack((np.ones(N), Y0, U0))
    W1 = np.dot(design_W1, para_set['mu_W1']) + np.random.normal(0, para_set['sigma_W1'], N)
    
    # 生成 U1, Y1
    # design matrix: [1, A1, Y0, U0]
    design_Y1U1 = np.column_stack((np.ones(N), A1, Y0, U0))
    Y1 = np.dot(design_Y1U1, para_set['mu_Y1']) + np.random.normal(0, para_set['sigma_Y1'], N)
    U1 = np.dot(design_Y1U1, para_set['mu_U1']) + np.random.normal(0, para_set['sigma_U1'], N)
    
    # 生成 A2
    # design matrix: [1, A1, Y0, U0, Y1, U1]
    design_A2 = np.column_stack((np.ones(N), A1, Y0, U0, Y1, U1))
    prop_score_2 = 1 / (1+np.exp(np.dot(design_A2, para_set['alpha_A2'])))
    A2_bin = np.random.binomial(1, prop_score_2, N)
    # map A2 from {0,1} to {-1,1}
    A2 = 2 * A2_bin - 1 
    # A2 = A2_bin
    
    # 生成 Z2
    # design matrix: [1, Z1, A2, Y1, U1, A1, Y0, U0]
    design_Z2 = np.column_stack((np.ones(N), Z1, A2, Y1, U1, A1, Y0, U0))
    Z2 = np.dot(design_Z2, para_set['mu_Z2']) + np.random.normal(0, para_set['sigma_Z2'], N)
    
    # 生成 W2
    # design matrix: [1, W1, Y1, U1, Y0, U0]
    design_W2 = np.column_stack((np.ones(N), W1, Y1, U1, Y0, U0))
    W2 = np.dot(design_W2, para_set['mu_W2']) + np.random.normal(0, para_set['sigma_W2'], N)
    
    # 生成 Y2
    # design matrix: [1, A2, A1, A2*A1, W2, W1, Y1, U1, Y0, U0]
    design_Y2 = np.column_stack((np.ones(N), A2, A1, A2 * A1, W2, W1, Y1, U1, Y0, U0))
    Y2 = np.dot(design_Y2, para_set['mu_Y2']) + np.random.normal(0, para_set['sigma_Y2'], N)
    
    # 组合成 DataFrame
    df = pd.DataFrame({
        'Y0': Y0, 'U0': U0, 'A1': A1, 'Z1': Z1, 'W1': W1,
        'Y1': Y1, 'U1': U1, 'A2': A2, 'Z2': Z2, 'W2': W2, 'Y2': Y2
    })
    return df

# 反事实数据生成函数 (Intervened Data Generation)
def intervened_data_gen(sample_size: int, para_set: dict, a: list = [1, 1]) -> pd.DataFrame:
    """
    生成在给定治疗序列 a (例如 [1, 1]) 下的反事实数据。
    这里的 A1 和 A2 不再是随机生成的，而是被强制设定为给定值。
    """
    N = sample_size
    # 生成 Y0, U0
    Y0 = np.random.normal(para_set['mu_Y0'], para_set['sigma_Y0'], N)
    U0 = np.random.normal(para_set['mu_U0'], para_set['sigma_U0'], N)
    
    # 强制设定 A1
    A1 = np.full(N, a[0])
    
    # 生成 Z1 (A1 固定)
    design_Z1 = np.column_stack((np.ones(N), A1, Y0, U0))
    Z1 = np.dot(design_Z1, para_set['mu_Z1']) + np.random.normal(0, para_set['sigma_Z1'], N)
    
    # 生成 W1
    design_W1 = np.column_stack((np.ones(N), Y0, U0))
    W1 = np.dot(design_W1, para_set['mu_W1']) + np.random.normal(0, para_set['sigma_W1'], N)
    
    # 生成 U1, Y1 (受固定的 A1 影响)
    design_Y1U1 = np.column_stack((np.ones(N), A1, Y0, U0))
    Y1 = np.dot(design_Y1U1, para_set['mu_Y1']) + np.random.normal(0, para_set['sigma_Y1'], N)
    U1 = np.dot(design_Y1U1, para_set['mu_U1']) + np.random.normal(0, para_set['sigma_U1'], N)
    
    # 强制设定 A2
    A2 = np.full(N, a[1])
    
    # 生成 Z2 (A2 固定)
    design_Z2 = np.column_stack((np.ones(N), Z1, A2, Y1, U1, A1, Y0, U0))
    Z2 = np.dot(design_Z2, para_set['mu_Z2']) + np.random.normal(0, para_set['sigma_Z2'], N)
    
    # 生成 W2
    design_W2 = np.column_stack((np.ones(N), W1, Y1, U1, Y0, U0))
    W2 = np.dot(design_W2, para_set['mu_W2']) + np.random.normal(0, para_set['sigma_W2'], N)
    
    # 生成 Y2 (反事实结果)
    design_Y2 = np.column_stack((np.ones(N), A2, A1, A2 * A1, W2, W1, Y1, U1, Y0, U0))
    Y2 = np.dot(design_Y2, para_set['mu_Y2']) + np.random.normal(0, para_set['sigma_Y2'], N)
    
    df = pd.DataFrame({
        'Y0': Y0, 'U0': U0, 'A1': A1, 'Z1': Z1, 'W1': W1,
        'Y1': Y1, 'U1': U1, 'A2': A2, 'Z2': Z2, 'W2': W2, 'Y2': Y2
    })
    return df

def dynamic_intervened_data_gen(sample_size: int, para_set: dict, f1=None, f2=None, device='cpu', seed=None) -> pd.DataFrame:
    """
    生成动态干预反事实数据。治疗 A1 和 A2 严格由传入的策略模型 f1 和 f2 动态决断。
    """
    if seed is not None:
        np.random.seed(seed)
    N = sample_size
    Y0 = np.random.normal(para_set['mu_Y0'], para_set['sigma_Y0'], N)
    U0 = np.random.normal(para_set['mu_U0'], para_set['sigma_U0'], N)
    
    with torch.no_grad():
        H1 = torch.tensor(Y0, dtype=torch.float32).unsqueeze(1).to(device)
        A1 = np.sign(f1(H1).cpu().numpy().flatten())
        A1[A1 == 0] = 1
        
    design_Z1 = np.column_stack((np.ones(N), A1, Y0, U0))
    Z1 = np.dot(design_Z1, para_set['mu_Z1']) + np.random.normal(0, para_set['sigma_Z1'], N)
    
    design_W1 = np.column_stack((np.ones(N), Y0, U0))
    W1 = np.dot(design_W1, para_set['mu_W1']) + np.random.normal(0, para_set['sigma_W1'], N)
    
    design_Y1U1 = np.column_stack((np.ones(N), A1, Y0, U0))
    Y1 = np.dot(design_Y1U1, para_set['mu_Y1']) + np.random.normal(0, para_set['sigma_Y1'], N)
    U1 = np.dot(design_Y1U1, para_set['mu_U1']) + np.random.normal(0, para_set['sigma_U1'], N)
    
    with torch.no_grad():
        H2 = torch.cat([
            torch.tensor(Y0, dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(Y1, dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(A1, dtype=torch.float32).unsqueeze(1).to(device)
        ], dim=1)
        A2 = np.sign(f2(H2).cpu().numpy().flatten())
        A2[A2 == 0] = 1
        
    design_Z2 = np.column_stack((np.ones(N), Z1, A2, Y1, U1, A1, Y0, U0))
    Z2 = np.dot(design_Z2, para_set['mu_Z2']) + np.random.normal(0, para_set['sigma_Z2'], N)
    
    design_W2 = np.column_stack((np.ones(N), W1, Y1, U1, Y0, U0))
    W2 = np.dot(design_W2, para_set['mu_W2']) + np.random.normal(0, para_set['sigma_W2'], N)
    
    design_Y2 = np.column_stack((np.ones(N), A2, A1, A2 * A1, W2, W1, Y1, U1, Y0, U0))
    Y2 = np.dot(design_Y2, para_set['mu_Y2']) + np.random.normal(0, para_set['sigma_Y2'], N)
    
    df = pd.DataFrame({
        'Y0': Y0, 'U0': U0, 'A1': A1, 'Z1': Z1, 'W1': W1,
        'Y1': Y1, 'U1': U1, 'A2': A2, 'Z2': Z2, 'W2': W2, 'Y2': Y2
    })
    return df


# Test for data generation
if __name__ == "__main__":
    adjusted_para_set = adjust_para_set_for_new_coding(origin_para_set)
    # 生成 x 条观测数据
    df_obs = data_gen(10000, adjusted_para_set)
    print("观测数据前5行:")
    print(df_obs.head())

    # 生成 1000 条反事实数据 (假设所有人都在 t=0 和 t=1 接受治疗)
    '''
    df_intervened = intervened_data_gen(1000, adjusted_para, a=[1, 1])
    print("\n反事实数据 (a=[1,1]) 前5行:")
    print(df_intervened.head())
    '''
    