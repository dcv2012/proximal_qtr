import numpy as np
import torch



def calculate_kernel_matrix(dataset, **kwargs):
    """
    计算数据集的高斯核矩阵。增加对极小样本量的健壮性检查。
    """
    n_samples = dataset.size(0)
    if n_samples < 2:
        return torch.eye(n_samples, device=dataset.device)
        
    K_X_euclidean = torch.cdist(dataset, dataset, p=2) ** 2
    triuInd = torch.triu_indices(n_samples, n_samples, offset=1)
    K_X_euclidean_upper = K_X_euclidean[triuInd[0], triuInd[1]]
    
    if K_X_euclidean_upper.numel() == 0:
        gamma = 1.0 # Fallback
    else:
        med = torch.quantile(K_X_euclidean_upper, 0.5)
        # 如果中位数为0（所有点重合），回退到 1.0 避免除零
        gamma = 1. / med if med > 0 else 1.0
        
    return torch.exp(-gamma * K_X_euclidean).squeeze()


def calculate_kernel_matrix_batched(dataset, batch_indices: tuple, **kwargs):
    """
    分批计算数据集的高斯核矩阵（针对大数据集优化）。
    
    参数:
    - dataset (torch.Tensor): 完整数据集，张量形状为 (n_samples, n_features)。
    - batch_indices (tuple): 批次索引，格式为 (start, end)，指定子集范围。
    - **kwargs: 额外关键字参数（当前未使用）。
    
    返回:
    - torch.Tensor: 批次核矩阵，形状为 (batch_size, n_samples)，表示批次样本与全数据集的相似度。
    
    功能:
    - 计算批次子集与全数据集间的欧几里得距离平方。
    - 使用非零距离的中位数作为 gamma(避免零距离影响)。
    - 应用高斯核函数: exp(-gamma * dist^2)，生成核矩阵。
    - 适用于内存受限场景，分批处理以减少计算开销。
    """
    
    start, end = batch_indices
    x = dataset[start:end]
    y = dataset
    dists = torch.cdist(x, y) ** 2
    nonzero_dists = dists[dists != 0]
    if nonzero_dists.numel() == 0:
        gamma = 1.0 # Fallback
    else:
        med = torch.median(nonzero_dists)
        gamma = 1 / med if med > 0 else 1.0
        
    return torch.exp(-gamma * dists).squeeze()

if __name__ == "__main__":
    # test the functions
    data = torch.randn(100, 10)
    K_full = calculate_kernel_matrix(data)
    K_batched = calculate_kernel_matrix_batched(data, (0, 100))
    print(K_full.shape)
    
# buyonggai