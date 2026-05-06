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
    if phi_type == 1:
        return 1.0 + (2.0 / np.pi) * torch.atan((np.pi * x) / 2.0)
    elif phi_type == 2:
        return 1.0 + x / (1.0 + torch.abs(x))
    elif phi_type == 3:
        return 1.0 + x / torch.sqrt(1.0 + x**2)
    elif phi_type == 4:
        return 1.0 + torch.tanh(x)
    else:
        raise ValueError("Invalid phi_type.")

def psi(x: torch.Tensor, y: torch.Tensor, phi_type: int = 1) -> torch.Tensor:
    if phi_type == 0:
        return torch.clamp(torch.min(x, y), max=1.0)
    return phi(x, phi_type) * phi(y, phi_type)

class Policy_Linear(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
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
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)

def prepare_outer_tensors(df: pd.DataFrame, ipw_weights: np.ndarray, q: float) -> TensorDataset:
    Y0 = torch.tensor(df['Y0'].values, dtype=torch.float32).unsqueeze(1)
    Y1 = torch.tensor(df['Y1'].values, dtype=torch.float32).unsqueeze(1)
    A1 = torch.tensor(df['A1'].values, dtype=torch.float32).unsqueeze(1)
    A2 = torch.tensor(df['A2'].values, dtype=torch.float32).unsqueeze(1)
    Y2 = torch.tensor(df['Y2'].values, dtype=torch.float32).unsqueeze(1)
    weights = torch.tensor(ipw_weights, dtype=torch.float32).unsqueeze(1)
    H1 = Y0
    H2 = torch.cat([Y0, Y1, A1], dim=1)
    indicator_Y2 = (Y2 > q).float()
    return TensorDataset(H1, H2, A1, A2, indicator_Y2, weights)

def train_outer_policies(train_loader: DataLoader, val_loader: DataLoader, params: Dict[str, Any], phi_type: int = 1, model_type: str = "linear") -> Tuple[nn.Module, nn.Module, float]:
    if model_type == "linear":
        model_f1 = Policy_Linear(1).to(device)
        model_f2 = Policy_Linear(3).to(device)
    elif model_type == "nn": 
        model_f1 = Policy_NN(1, hidden_dim=params.get('network_width', 32), num_layers=params.get('network_depth', 2)).to(device)
        model_f2 = Policy_NN(3, hidden_dim=params.get('network_width', 32), num_layers=params.get('network_depth', 2)).to(device)
    else:
        raise ValueError("Invalid model_type")
    
    optimizer = optim.AdamW(list(model_f1.parameters()) + list(model_f2.parameters()), lr=params['lr'], weight_decay=params['l2'])
    best_val_loss = float('inf')
    best_f1_state = model_f1.state_dict()
    best_f2_state = model_f2.state_dict()
    patience = 20
    counter = 0
    
    for epoch in range(params['epochs']):
        model_f1.train()
        model_f2.train()
        for batch in train_loader:
            b_H1, b_H2, b_A1, b_A2, b_I, b_W = [t.to(device) for t in batch]
            optimizer.zero_grad()
            f1_pred = model_f1(b_H1)
            f2_pred = model_f2(b_H2)
            psi_val = psi(b_A1 * f1_pred, b_A2 * f2_pred, phi_type)
            loss = -torch.mean(b_I * b_W * psi_val)
            loss.backward()
            optimizer.step()
            
        model_f1.eval()
        model_f2.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                b_H1, b_H2, b_A1, b_A2, b_I, b_W = [t.to(device) for t in batch]
                f1_pred = model_f1(b_H1)
                f2_pred = model_f2(b_H2)
                psi_val = psi(b_A1 * f1_pred, b_A2 * f2_pred, phi_type)
                total_val_loss += (-torch.mean(b_I * b_W * psi_val)).item()
            total_val_loss /= len(val_loader)
            
        if not np.isnan(total_val_loss) and total_val_loss < best_val_loss:
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

def optimize_outer_hyperparams(df_train: pd.DataFrame, ipw_train: np.ndarray, df_val: pd.DataFrame, ipw_val: np.ndarray, q: float, n_trials: int = 15, epochs: int = 200, phi_type: int = 1, model_type: str = "linear") -> Dict[str, Any]:
    train_dataset = prepare_outer_tensors(df_train, ipw_train, q)
    val_dataset = prepare_outer_tensors(df_val, ipw_val, q)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    def objective(trial):
        params = {'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True), 'l2': trial.suggest_float('l2', 1e-8, 1e-6, log=True), 'epochs': epochs}
        if model_type == "nn":
            params['network_width'] = trial.suggest_categorical('network_width', [32, 64, 128])
            params['network_depth'] = trial.suggest_int('network_depth', 2, 4)
        _, _, val_loss = train_outer_policies(train_loader, val_loader, params, phi_type, model_type)
        return val_loss
        
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=torch.randint(0, 100000, (1,)).item()))
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_params['epochs'] = epochs
    return best_params
