import numpy as np

def compute_weighted_quantile(y: np.ndarray, weights: np.ndarray, tau: float) -> float:
    y = np.asarray(y)
    weights = np.asarray(weights)
    
    mask = weights > 1e-12
    y_filtered = y[mask]
    w_filtered = weights[mask]
    
    if len(y_filtered) == 0:
        return 0.0
        
    sort_idx = np.argsort(y_filtered)
    y_sorted = y_filtered[sort_idx]
    w_sorted = w_filtered[sort_idx]
    
    w_sum = np.sum(w_sorted)
    if w_sum == 0:
        return 0.0
        
    w_norm = w_sorted / w_sum
    cum_weights = np.cumsum(w_norm)
    
    idx = np.searchsorted(cum_weights, tau)
    idx = min(idx, len(y_sorted) - 1)
    
    return float(y_sorted[idx])

def inner_optimization(Y2: np.ndarray, A1: np.ndarray, A2: np.ndarray, d1_pred: np.ndarray, d2_pred: np.ndarray, weights_ipw: np.ndarray, tau: float = 0.5) -> float:
    """
    SRA 估计器的内层优化：使用 IPW 权重求解加权分位数 q。
    """
    Y2 = np.asarray(Y2)
    A1 = np.asarray(A1)
    A2 = np.asarray(A2)
    d1_pred = np.asarray(d1_pred)
    d2_pred = np.asarray(d2_pred)
    weights_ipw = np.asarray(weights_ipw)
    
    match_indicator = (d1_pred == A1) & (d2_pred == A2)
    final_weights = weights_ipw * match_indicator
    
    return compute_weighted_quantile(Y2, final_weights, tau)

def inner_optimization_grid(Y2: np.ndarray, ipw_vals: np.ndarray, phi1: np.ndarray, phi2: np.ndarray, grid_Q: np.ndarray, tau: float = 0.5):
    """
    SRA AO 的内层优化：网格搜索寻找分位数 q。
    SRA 的 IPW 目标函数使用原始统计量（不做 Hajek 自归一化）。
    """
    from typing import Tuple
    Y2 = np.asarray(Y2)
    ipw_vals = np.asarray(ipw_vals)
    phi1 = np.asarray(phi1)
    phi2 = np.asarray(phi2)
    grid_Q = np.asarray(grid_Q)
    
    ipw_phi_prod = ipw_vals * phi1 * phi2
    norm_factor = np.mean(ipw_phi_prod) + 1e-10
    
    # 使用 numpy broadcast 高效计算所有 grid 点的生存值
    sv_array = np.mean((Y2[:, None] > grid_Q[None, :]) * ipw_phi_prod[:, None], axis=0)
    
    best_idx = np.argmin(np.abs(sv_array - (1 - tau)))
    q_new = float(grid_Q[best_idx])
    sv_val = float(sv_array[best_idx])
    
    return q_new, sv_val
