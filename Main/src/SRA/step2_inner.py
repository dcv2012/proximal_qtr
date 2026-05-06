import numpy as np
import torch

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


def inner_optimization_grid(
    Y2,
    ipw_vals,
    phi1,
    phi2,
    grid_Q,
    tau: float = 0.5,
    device: torch.device = _DEVICE,
):
    """
    SRA AO 的内层优化：网格搜索寻找分位数 q（全程在 GPU 上运行）。
    SRA 的 IPW 目标函数使用 argmin 策略（对称 IPW 估计）。
    """
    # 上传到 GPU（如果已是 tensor 则零拷贝）
    Y2_t     = torch.as_tensor(Y2,      dtype=torch.float32, device=device)
    ipw_t    = torch.as_tensor(ipw_vals, dtype=torch.float32, device=device)
    phi1_t   = torch.as_tensor(phi1,    dtype=torch.float32, device=device)
    phi2_t   = torch.as_tensor(phi2,    dtype=torch.float32, device=device)
    grid_Q_t = torch.as_tensor(grid_Q,  dtype=torch.float32, device=device)

    weights  = ipw_t * phi1_t * phi2_t           # (N,)

    # [N, G] 广播：全程在 GPU
    mask     = (Y2_t.unsqueeze(1) > grid_Q_t.unsqueeze(0)).float()
    sv_array = (mask * weights.unsqueeze(1)).mean(dim=0)   # (G,)

    best_idx = int(torch.argmin(torch.abs(sv_array - (1.0 - tau))).item())
    q_new    = float(grid_Q[best_idx] if isinstance(grid_Q, np.ndarray) else grid_Q_t[best_idx].item())
    sv_val   = float(sv_array[best_idx].item())

    return q_new, sv_val
