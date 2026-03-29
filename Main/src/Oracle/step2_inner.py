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

def inner_optimization(Y2: np.ndarray, A1: np.ndarray, A2: np.ndarray, d1_pred: np.ndarray, d2_pred: np.ndarray, oracle_weights: np.ndarray, tau: float = 0.5) -> float:
    """
    Oracle 估计器的内层优化：使用真实的 Oracle IPW 权重求解加权分位数 q。
    """
    Y2 = np.asarray(Y2)
    A1 = np.asarray(A1)
    A2 = np.asarray(A2)
    d1_pred = np.asarray(d1_pred)
    d2_pred = np.asarray(d2_pred)
    oracle_weights = np.asarray(oracle_weights)
    match_indicator = (d1_pred == A1) & (d2_pred == A2)
    final_weights = oracle_weights * match_indicator
    return compute_weighted_quantile(Y2, final_weights, tau)
