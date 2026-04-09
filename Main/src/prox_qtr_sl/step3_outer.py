import torch
from typing import Dict, Any, Tuple
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import optuna

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def phi(x: torch.Tensor, phi_type: int = 1) -> torch.Tensor:
    r"""
    平滑替代函数，可选择不同的平滑形式拟合指示函数。
    1: \phi(x) = 1 + (2/\pi) * \arctan(\pi * x / 2)
    2: \phi(x) = 1 + x / (1 + |x|)
    3: \phi(x) = 1 + x / \sqrt{1 + x^2}
    4: \phi(x) = 1 + \tanh(x)
    """
    if phi_type == 1:
        return 1.0 + (2.0 / np.pi) * torch.atan((np.pi * x) / 2.0)
    elif phi_type == 2:
        return 1.0 + x / (1.0 + torch.abs(x))
    elif phi_type == 3:
        return 1.0 + x / torch.sqrt(1.0 + x**2)
    elif phi_type == 4:
        return 1.0 + torch.tanh(x)
    else:
        raise ValueError("Invalid phi_type. Must be 1, 2, 3, or 4.")

def psi(x: torch.Tensor, y: torch.Tensor, phi_type: int = 1) -> torch.Tensor:
    r"""
    二维替代函数: \psi(x, y) = \phi(x) * \phi(y)
    """
    if phi_type == 0:
        return torch.clamp(torch.min(x, y), max=1.0)
    return phi(x, phi_type) * phi(y, phi_type)

class Policy_Linear(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
        # 初始化权重使得输出不会在一开始过度饱和
        nn.init.xavier_normal_(self.linear.weight)
        
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)

class Policy_NN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 2) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.1))
            in_dim = hidden_dim
            
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)

def prepare_outer_tensors(df: pd.DataFrame, q22_vals: np.ndarray, q: float) -> TensorDataset:
    """
    构建外层策略优化的 PyTorch DataLoader 和 Dataset
    """
    Y0 = torch.tensor(df['Y0'].values, dtype=torch.float32).unsqueeze(1)
    Y1 = torch.tensor(df['Y1'].values, dtype=torch.float32).unsqueeze(1)
    A1 = torch.tensor(df['A1'].values, dtype=torch.float32).unsqueeze(1)
    A2 = torch.tensor(df['A2'].values, dtype=torch.float32).unsqueeze(1)
    Y2 = torch.tensor(df['Y2'].values, dtype=torch.float32).unsqueeze(1)
    
    q22 = torch.tensor(q22_vals, dtype=torch.float32).unsqueeze(1)
    
    # 构建 H1, H2
    H1 = Y0
    H2 = torch.cat([Y0, Y1, A1], dim=1)
    
    # 目标生存概率的指示器 I(Y2 > q)
    indicator_Y2 = (Y2 > q).float()
    
    return TensorDataset(H1, H2, A1, A2, indicator_Y2, q22)


def train_outer_policies(train_loader: DataLoader, val_loader: DataLoader, params: Dict[str, Any], phi_type: int = 1, model_type: str = "linear") -> Tuple[nn.Module, nn.Module, float]:
    """
    给定参数下的策略模型联合训练函数。
    允许通过 model_type 选择 f1 和 f2 为 linear 或 nn。
    """
    if model_type == "linear":
        model_f1 = Policy_Linear(input_dim=1).to(device)
        model_f2 = Policy_Linear(input_dim=3).to(device)
    elif model_type == "nn":
        model_f1 = Policy_NN(input_dim=1, hidden_dim=params.get('network_width', 32), num_layers=params.get('network_depth', 2)).to(device)
        model_f2 = Policy_NN(input_dim=3, hidden_dim=params.get('network_width', 32), num_layers=params.get('network_depth', 2)).to(device)
    else:
        raise ValueError("model_type must be either 'linear' or 'nn'.")
    
    # 分离优化器以支持坐标下降 (Coordinate Descent)
    optimizer_f1 = optim.AdamW(model_f1.parameters(), lr=params['lr'], weight_decay=params['l2'])
    optimizer_f2 = optim.AdamW(model_f2.parameters(), lr=params['lr'], weight_decay=params['l2'])
                           
    best_val_loss = float('inf')
    best_f1_state = None
    best_f2_state = None
    
    # 早停逻辑
    patience = 20
    counter = 0
    
    for epoch in range(params['epochs']):
        model_f1.train()
        model_f2.train()
        
        for batch in train_loader:
            b_H1, b_H2, b_A1, b_A2, b_I, b_q22 = [t.to(device) for t in batch]
            
            # --- 坐标下降 Step 1: 更新 f1 (冻结 f2) ---
            optimizer_f1.zero_grad()
            f1_curr = model_f1(b_H1)
            with torch.no_grad():
                f2_fixed = model_f2(b_H2)
            
            psi_val_1 = psi(b_A1 * f1_curr, b_A2 * f2_fixed, phi_type)
            loss_f1 = -torch.mean(b_I * b_q22 * psi_val_1)
            loss_f1.backward()
            # torch.nn.utils.clip_grad_norm_(model_f1.parameters(), max_norm=1.0)
            optimizer_f1.step()

            # --- 坐标下降 Step 2: 更新 f2 (冻结 f1) ---
            optimizer_f2.zero_grad()
            f2_curr = model_f2(b_H2)
            with torch.no_grad():
                f1_fixed = model_f1(b_H1)
                
            psi_val_2 = psi(b_A1 * f1_fixed, b_A2 * f2_curr, phi_type)
            loss_f2 = -torch.mean(b_I * b_q22 * psi_val_2)
            loss_f2.backward()
            # torch.nn.utils.clip_grad_norm_(model_f2.parameters(), max_norm=1.0)
            optimizer_f2.step()
            
        # Validation Loop
        model_f1.eval()
        model_f2.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                b_H1, b_H2, b_A1, b_A2, b_I, b_q22 = [t.to(device) for t in batch]
                f1_pred = model_f1(b_H1)
                f2_pred = model_f2(b_H2)
            
                psi_val = psi(b_A1 * f1_pred, b_A2 * f2_pred, phi_type)
                objective = torch.mean(b_I * b_q22 * psi_val)
                total_val_loss += (-objective).item()
                
            total_val_loss /= len(val_loader)
            
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_f1_state = model_f1.state_dict()
            best_f2_state = model_f2.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
                
    if best_f1_state is not None:
        model_f1.load_state_dict(best_f1_state)
        model_f2.load_state_dict(best_f2_state)
        
    return model_f1, model_f2, best_val_loss


def optimize_outer_hyperparams(df_train: pd.DataFrame, q22_train: np.ndarray, df_val: pd.DataFrame, q22_val: np.ndarray, q: float, n_trials: int = 10, epochs: int = 200, phi_type: int = 1, model_type: str = "linear") -> Dict[str, Any]:
    """
    使用 Optuna 优化外层模型(f1, f2)的架构与学习率
    """
    train_dataset = prepare_outer_tensors(df_train, q22_train, q)
    val_dataset = prepare_outer_tensors(df_val, q22_val, q)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    def objective(trial):
        params = {
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            'l2': trial.suggest_float('l2', 1e-8, 1e-6, log=True),
            'epochs': epochs
        }
        if model_type == "nn":
            params['network_width'] = trial.suggest_categorical('network_width', [32, 64, 128])
            params['network_depth'] = trial.suggest_int('network_depth', 2, 4)
            
        _, _, best_val_loss = train_outer_policies(train_loader, val_loader, params, phi_type, model_type)
        # Optuna seeks to minimize objective, which matches our negative survival function
        return best_val_loss
        
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_params['epochs'] = epochs 
    return best_params
