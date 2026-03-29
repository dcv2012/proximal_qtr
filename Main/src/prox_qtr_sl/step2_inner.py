import numpy as np

def compute_weighted_quantile(y: np.ndarray, weights: np.ndarray, tau: float) -> float:
    r"""
    计算加权分位数，求解加权 check function 最小化问题:
    \arg\min_q \sum w_i * \rho_\tau(y_i - q)
    该问题在统计学中等价于求经验分布的加权 \tau 分位数。
    
    y: array-like of shape (N,)
    weights: array-like of shape (N,), weights should be >= 0
    tau: float in (0, 1)
    """
    y = np.asarray(y)
    weights = np.asarray(weights)
    
    # 除去权重为0的样本以加速排序和累加
    mask = weights > 1e-12
    y_filtered = y[mask]
    w_filtered = weights[mask]
    
    if len(y_filtered) == 0:
        return 0.0 # 极端情况：所有样本权重均为0
        
    # 对 Y 值进行排序
    sort_idx = np.argsort(y_filtered)
    y_sorted = y_filtered[sort_idx]
    w_sorted = w_filtered[sort_idx]
    
    # 权重归一化
    w_sum = np.sum(w_sorted)
    if w_sum == 0:
        return 0.0
        
    w_norm = w_sorted / w_sum
    cum_weights = np.cumsum(w_norm)
    
    # 寻找累积权重大于等于 tau 的第一个元素
    idx = np.searchsorted(cum_weights, tau)
    # 防止索引越界
    idx = min(idx, len(y_sorted) - 1)
    
    return float(y_sorted[idx])

def inner_optimization(Y2: np.ndarray, A1: np.ndarray, A2: np.ndarray, d1_pred: np.ndarray, d2_pred: np.ndarray, q22_vals: np.ndarray, tau: float = 0.5) -> float:
    r"""
    给定策略下的内层优化，求解最优分位数边界 q。
    该实现按照如下数学公式求解:
    \arg\min_q \sum \{ \hat{q}_{22,i} * I(d_1 = A_1) * I(d_2 = A_2) * \rho_\tau(Y_2 - q) \}
    
    Y2: 最终观测结果 (N,)
    A1: 阶段一实际观测治疗方案 (N,)
    A2: 阶段二实际观测治疗方案 (N,)
    d1_pred: \hat{d}_1 阶段一由模型 f1 决策出的符号 (N,)
    d2_pred: \hat{d}_2 阶段二由模型 f2 决策出的符号 (N,)
    q22_vals: Step 1 预估计出的 \hat{q}_{22} 值 (N,)
    tau: 目标分位数值 (默认 0.5)
    """
    # 转换所有输入为 Numpy 数组，保证行为一致
    Y2 = np.asarray(Y2)
    A1 = np.asarray(A1)
    A2 = np.asarray(A2)
    d1_pred = np.asarray(d1_pred)
    d2_pred = np.asarray(d2_pred)
    q22_vals = np.asarray(q22_vals)
    
    # 计算综合权重：预估的大样本权重 * 指示函数的精准匹配
    match_indicator = (d1_pred == A1) & (d2_pred == A2)
    weights = q22_vals * match_indicator
    
    optimal_q = compute_weighted_quantile(Y2, weights, tau)
    return optimal_q
