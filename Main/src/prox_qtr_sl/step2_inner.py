import numpy as np
import torch
from typing import Dict, List, Tuple

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def inner_optimization_grid(
    Y2: np.ndarray,
    q22_vals: np.ndarray,
    phi1: np.ndarray,
    phi2: np.ndarray,
    grid_Q: np.ndarray,
    tau: float = 0.5,
    device: torch.device = _DEVICE,
) -> Tuple[float, float]:
    r"""
    给定当前预估出的策略 (转化为平滑响应 phi)，利用网格搜索(Grid Search) 寻找满足
    生存约束的最大分位数 q:
        q^(k) = sup { q_s in Q : SV_hat(q_s) >= 1 - tau }

    全程在 GPU (或 CPU fallback) 上通过 torch 广播完成，避免 CPU 瓶颈。

    Y2:       最终观测结果 (N,)
    q22_vals: Step 1 预估计出的 hat{q}_{22} 值 (N,)
    phi1:     阶段一的策略响应指示函数 (N,)
    phi2:     阶段二的策略响应指示函数 (N,)
    grid_Q:   一维搜索网格 (升序有序的 Y2 唯一值)
    tau:      目标分位数值 (默认 0.5)
    device:   运算设备 (默认 CUDA if available)

    Returns:
        最优边界 q_new 和 它对应的生存值 sv_val
    """
    # 上传到 GPU（如果已经是 tensor 则直接转设备，否则从 numpy 转）
    Y2_t      = torch.as_tensor(Y2,      dtype=torch.float32, device=device)
    q22_t     = torch.as_tensor(q22_vals, dtype=torch.float32, device=device)
    phi1_t    = torch.as_tensor(phi1,    dtype=torch.float32, device=device)
    phi2_t    = torch.as_tensor(phi2,    dtype=torch.float32, device=device)
    grid_Q_t  = torch.as_tensor(grid_Q,  dtype=torch.float32, device=device)

    # 提前连乘权重  shape: (N,)
    weights = q22_t * phi1_t * phi2_t

    # [N, len(grid_Q)] 广播：全程在 GPU
    # mask[i, j] = 1 iff Y2[i] > grid_Q[j]
    mask = (Y2_t.unsqueeze(1) > grid_Q_t.unsqueeze(0)).float()   # (N, G)
    sv_array = (mask * weights.unsqueeze(1)).mean(dim=0)           # (G,)

    target = 1.0 - tau
    feasible_mask = sv_array >= target
    feasible_idx  = feasible_mask.nonzero(as_tuple=False).squeeze(1)

    if feasible_idx.numel() > 0:
        # 严格对应理论上确界: sup { q in Q : SV(q) >= 1-tau }
        # grid_Q 是升序的，最后一个可行索引对应最大的 q
        best_idx = int(feasible_idx[-1].item())
        print(f"    -> feasible set has {feasible_idx.numel()} points, "
              f"select rightmost idx={best_idx}, q={grid_Q[best_idx]:.4f}")
    else:
        # 空集 fallback: 退化为最近点 argmin
        best_idx = int(torch.argmin(torch.abs(sv_array - target)).item())
        print(f"    -> [WARN] feasible set is EMPTY "
              f"(sv_max={sv_array.max().item():.4f} < target={target:.4f}). "
              f"Falling back to argmin, idx={best_idx}, q={grid_Q[best_idx]:.4f}. "
              f"Check q22 estimation.")

    q_new  = float(grid_Q[best_idx])
    sv_val = float(sv_array[best_idx].item())

    return q_new, sv_val



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
