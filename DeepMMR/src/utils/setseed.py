import random
import numpy as np
import torch

def set_seed(seed=20048):
    # 1. Python 内置随机模块
    random.seed(seed)
    # 2. NumPy 随机数
    np.random.seed(seed)
    # 3. PyTorch CPU 随机数
    torch.manual_seed(seed)
    # 4. PyTorch GPU 随机数 (如果有多个 GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 5. 确保 CuDNN 算法的确定性 (极其重要，但可能会稍微牺牲一点训练速度)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
