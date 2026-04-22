import torch
from Main.src.deepmmr.kernel_utils import calculate_kernel_matrix_batched



def MMR_loss(model_output, target, kernel_matrix, loss_name: str, lambda_reg: float = 0.0):  # batch_indices=None:
    """
    计算 MMR 损失函数。
    
    参数:
    - model_output (torch.Tensor): 模型输出，张量形状为 (n_samples,) 或 (n_samples, 1)。
    - target (torch.Tensor): 目标值，张量形状与 model_output 相同。
    - kernel_matrix (torch.Tensor): 高斯核矩阵，形状为 (n_samples, n_samples)。
    - loss_name (str): 损失类型，必须为 'U_statistic' 或 'V_statistic'。
    
    返回:
    - torch.Tensor: 损失值，标量。
    
    功能:
    - 计算残差(model_output - target)。
    - 根据 loss_name 计算 U-statistic(排除对角线)或 V-statistic(包括对角线)的核加权残差平方和。
    - U-statistic 适用于无偏估计, V-statistic 适用于方差估计。
    """
    residual = model_output - target
    n = residual.shape[0]
    K = kernel_matrix.clone()

    if loss_name == "U_statistic":
        # calculate U statistic (see Serfling 1980)
        K.fill_diagonal_(0)
        loss = (residual.T @ K @ residual) / (n * (n-1))
    elif loss_name == "V_statistic":
        # calculate V statistic (see Serfling 1980)
        loss = (residual.T @ K @ residual) / (n ** 2)
    else:
        raise ValueError(f"{loss_name} is not valid. Must be 'U_statistic' or 'V_statistic'.")
    
    if lambda_reg > 0.0:
        loss = loss + lambda_reg * torch.mean(model_output ** 2)
    
    return loss
        

def MMR_loss_batched(model_output, propensity_score, kernel_inputs, kernel, batch_size: int, loss_name: str):
    """
    分批计算 MMR 损失函数（针对大数据集优化）。
    
    参数:
    - model_output (torch.Tensor): 模型输出，张量形状为 (n_samples,) 或 (n_samples, 1)。
    - propensity_score (torch.Tensor): 倾向得分，张量形状与 model_output 相同。
    - kernel_inputs (torch.Tensor): 核输入数据，张量形状为 (n_samples, n_features)。
    - kernel: 核函数（当前未使用，保留兼容性）。
    - batch_size (int): 批次大小，用于分批计算。
    - loss_name (str): 损失类型，必须为 'U_statistic' 或 'V_statistic'。
    
    返回:
    - float: 累计损失值。
    
    功能:
    - 计算加权残差(model_output * propensity_score - 1)。
    - 分批计算核矩阵，使用 calculate_kernel_matrix_batched。
    - 根据 loss_name 累加 U-statistic 或 V-statistic 损失，适用于内存受限场景。
    """
    
    residual = model_output * propensity_score - 1
    n = residual.shape[0]

    loss = 0
    for i in range(0, n, batch_size):
        partial_kernel_matrix = calculate_kernel_matrix_batched(kernel_inputs, (i, i+batch_size), kernel)
        if loss_name == "V_statistic":
            factor = n ** 2
        if loss_name == "U_statistic":
            factor = n * (n-1)
            # zero out the main diagonal of the full matrix
            for row_idx in range(partial_kernel_matrix.shape[0]):
                partial_kernel_matrix[row_idx, row_idx+i] = 0
        temp_loss = residual[i:(i+batch_size)].T @ partial_kernel_matrix @ residual / factor
        loss += temp_loss[0, 0]
    return loss

# buyonggai