import numpy as np
from typing import Tuple

def inner_optimization_grid(Y2: np.ndarray, q22_vals: np.ndarray, phi1: np.ndarray, phi2: np.ndarray, grid_Q: np.ndarray, tau: float = 0.5) -> Tuple[float, float]:
    r"""
    给定当前预估出的策略 (转化为平滑响应 phi)，利用网格搜索(Grid Search) 寻找使得
    生存估算函数最接近 (1 - \tau) 的最优分位数 q。
    该实现按照如下数学公式求解:
    \arg\min_q | \frac{1}{n} \sum \{ \hat{q}_{22,i} * I(Y_{2,i} > q) * \Phi_1 * \Phi_2 \} - (1 - \tau) |
    
    Y2: 最终观测结果 (N,)
    q22_vals: Step 1 预估计出的 \hat{q}_{22} 值 (N,)
    phi1: 阶段一的平滑策略响应 \Phi\{A_1 f_1 / h_n\} (N,)
    phi2: 阶段二的平滑策略响应 \Phi\{A_2 f_2 / h_n\} (N,)
    grid_Q: 一维搜索网格 (比如有序独特的 Y2)
    tau: 目标分位数值 (默认 0.5)
    
    Returns:
        最优边界 q_new 和 它对应的生存值 sv_val
    """
    Y2 = np.asarray(Y2)
    q22_vals = np.asarray(q22_vals)
    phi1 = np.asarray(phi1)
    phi2 = np.asarray(phi2)
    grid_Q = np.asarray(grid_Q)
    
    # 提前连乘权重加速广播运算
    q22_phi_prod = q22_vals * phi1 * phi2
    
    # 使用 numpy broadcast 高效计算所有 grid 点的生存值.
    # (Y2[:, None] > grid_Q[None, :]) 返回 [n_samples, len(grid_Q)] 的 bool mask.
    sv_array = np.mean((Y2[:, None] > grid_Q[None, :]) * q22_phi_prod[:, None], axis=0)
    
    best_idx = np.argmin(np.abs(sv_array - (1 - tau)))
    q_new = grid_Q[best_idx]
    sv_val = sv_array[best_idx]
    
    return float(q_new), float(sv_val)
