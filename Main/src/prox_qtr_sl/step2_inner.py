import numpy as np
from typing import Dict, List, Tuple

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
    norm_factor = np.mean(q22_phi_prod)
    
    # 使用 numpy broadcast 高效计算所有 grid 点的生存值.
    # 引入 Hajek Estimator 自归一化机制，修正不完美的经验均值，严格限定概率界限在 [0,1] 内
    # (Y2[:, None] > grid_Q[None, :]) 返回 [n_samples, len(grid_Q)] 的 bool mask.
    raw_sv_array = np.mean((Y2[:, None] > grid_Q[None, :]) * q22_phi_prod[:, None], axis=0)
    sv_array = raw_sv_array / (norm_factor + 1e-10)
    
    # no-Hajek
    # sv_array = raw_sv_array
    
    '''
    target = 1 - tau
    feasible_idx = np.where(sv_array >= target)[0]
    if feasible_idx.size > 0:
        # 选择最右侧可行根，避免落入低 q 的局部根
        best_idx = feasible_idx[-1]
    else:
        best_idx = np.argmin(np.abs(sv_array - target))
    '''
    
    best_idx = np.argmin(np.abs(sv_array - (1 - tau)))
    q_new = grid_Q[best_idx]
    sv_val = sv_array[best_idx]
    
    return float(q_new), float(sv_val)


def compute_sv_curve_on_grid(
    Y2: np.ndarray,
    q22_vals: np.ndarray,
    phi1: np.ndarray,
    phi2: np.ndarray,
    grid_Q: np.ndarray,
    tau: float = 0.5,
) -> Dict[str, object]:
    r"""
    计算给定策略下整条网格上的经验生存曲线 \hat{SV}_\Phi(q)，并返回与目标线 (1-\tau) 的交点信息。

    该函数用于诊断/可视化，不改变原始 inner_optimization_grid 的行为。
    """
    Y2 = np.asarray(Y2)
    q22_vals = np.asarray(q22_vals)
    phi1 = np.asarray(phi1)
    phi2 = np.asarray(phi2)
    grid_Q = np.asarray(grid_Q)

    q22_phi_prod = q22_vals * phi1 * phi2
    norm_factor = np.mean(q22_phi_prod)
    raw_sv_array = np.mean((Y2[:, None] > grid_Q[None, :]) * q22_phi_prod[:, None], axis=0)
    sv_array = raw_sv_array / (norm_factor + 1e-10)

    target = 1 - tau
    feasible_idx = np.where(sv_array >= target)[0]
    if feasible_idx.size > 0:
        best_idx = int(feasible_idx[-1])
    else:
        best_idx = int(np.argmin(np.abs(sv_array - target)))

    crossing_qs: List[float] = []
    for i in range(len(grid_Q) - 1):
        y1 = sv_array[i] - target
        y2 = sv_array[i + 1] - target
        if y1 == 0:
            crossing_qs.append(float(grid_Q[i]))
        if y1 * y2 < 0:
            x1 = float(grid_Q[i])
            x2 = float(grid_Q[i + 1])
            # 线性插值近似交点
            q_cross = x1 - y1 * (x2 - x1) / (y2 - y1)
            crossing_qs.append(float(q_cross))
    if len(grid_Q) > 0 and (sv_array[-1] - target) == 0:
        crossing_qs.append(float(grid_Q[-1]))

    return {
        "grid_Q": grid_Q,
        "sv_array": sv_array,
        "target": float(target),
        "best_idx": best_idx,
        "q_best": float(grid_Q[best_idx]),
        "sv_best": float(sv_array[best_idx]),
        "norm_factor": float(norm_factor),
        "crossing_qs": crossing_qs,
    }
