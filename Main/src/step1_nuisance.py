import sys
import os
from typing import Dict, Any, Tuple, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import optuna
from pathlib import Path

# 从本地 deepmmr 目录引用模型和损失函数
from Main.src.deepmmr.MMR_model import MLP_for_MMR
from Main.src.deepmmr.MMR_loss import MMR_loss
from Main.src.deepmmr.kernel_utils import calculate_kernel_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EarlyStopping:
    def __init__(self, patience: int = 15, delta: float = 1e-4) -> None:
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def prepare_tensors(df: pd.DataFrame, a1: int, a2: int) -> TensorDataset:
    """
    仅筛选 A1 == a1 的数据，避免不必要的网络结构训练。
    提取出张量用于 q11 和 q22 的训练。
    """
    df_filtered = df[df['A1'] == a1].copy()
    
    Y0 = torch.tensor(df_filtered['Y0'].values, dtype=torch.float32).unsqueeze(1)
    Z1 = torch.tensor(df_filtered['Z1'].values, dtype=torch.float32).unsqueeze(1)
    W1 = torch.tensor(df_filtered['W1'].values, dtype=torch.float32).unsqueeze(1)
    
    # 因为 A1 已经被筛选成了常数, 处理 tt1(目标指示器) 时实际上始终是 1。
    # 这里保持逻辑，对于 q11: I(A1=a1)
    tt1 = torch.tensor((df_filtered['A1'] == a1).values, dtype=torch.float32).unsqueeze(1)
    
    Y1 = torch.tensor(df_filtered['Y1'].values, dtype=torch.float32).unsqueeze(1)
    Z2 = torch.tensor(df_filtered['Z2'].values, dtype=torch.float32).unsqueeze(1)
    W2 = torch.tensor(df_filtered['W2'].values, dtype=torch.float32).unsqueeze(1)
    
    # tt2: I(A2=a2)
    tt2 = torch.tensor((df_filtered['A2'] == a2).values, dtype=torch.float32).unsqueeze(1)
    
    return TensorDataset(Z1, Y0, W1, tt1, Z2, Y1, W2, tt2)


def train_q11(train_loader: DataLoader, val_loader: DataLoader, params: Dict[str, Any]) -> Tuple[nn.Module, float]:
    input_dim = 2 # Z1 (1), Y0 (1)
    model = MLP_for_MMR(input_dim, params).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['l2'])
    
    early_stopping = EarlyStopping()
    best_model_state = None
    best_val_loss = float('inf')
    
    for epoch in range(params['epochs']):
        model.train()
        for batch in train_loader:
            Z1, Y0, W1, tt1, _, _, _, _ = [t.to(device) for t in batch]
            
            optimizer.zero_grad()
            pred = model(torch.cat([Z1, Y0], dim=1))
            
            kernel_inputs = torch.cat([W1, Y0], dim=1)
            kernel_matrix = calculate_kernel_matrix(kernel_inputs)
            
            loss = torch.abs(MMR_loss(pred * tt1, torch.ones_like(pred), kernel_matrix, loss_name='U_statistic'))
            loss.backward()
            optimizer.step()
            
        # Eval
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for batch in val_loader:
                Z1, Y0, W1, tt1, _, _, _, _ = [t.to(device) for t in batch]
                pred = model(torch.cat([Z1, Y0], dim=1))
                kernel_matrix = calculate_kernel_matrix(torch.cat([W1, Y0], dim=1))
                v_loss = torch.abs(MMR_loss(pred * tt1, torch.ones_like(pred), kernel_matrix, loss_name='U_statistic'))
                total_val_loss += v_loss.item()
            total_val_loss /= len(val_loader)
            
        early_stopping(total_val_loss)
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_model_state = model.state_dict()
        if early_stopping.early_stop:
            break
            
    model.load_state_dict(best_model_state)
    return model, best_val_loss

def train_q22(train_loader: DataLoader, val_loader: DataLoader, model_q11: nn.Module, params: Dict[str, Any]) -> Tuple[nn.Module, float]:
    input_dim = 4 # Z1 (1), Z2 (1), Y0 (1), Y1 (1)
    model = MLP_for_MMR(input_dim, params).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['l2'])
    
    early_stopping = EarlyStopping()
    best_model_state = None
    best_val_loss = float('inf')
    
    model_q11.eval()
    
    for epoch in range(params['epochs']):
        model.train()
        for batch in train_loader:
            Z1, Y0, W1, tt1, Z2, Y1, W2, tt2 = [t.to(device) for t in batch]
            
            optimizer.zero_grad()
            
            # Predict q11 (Cached dynamically per batch, no gradient)
            with torch.no_grad():
                q11_pred = model_q11(torch.cat([Z1, Y0], dim=1))
            
            # Predict q22
            pred2 = model(torch.cat([Z1, Z2, Y0, Y1], dim=1))
            kernel_inputs2 = torch.cat([W1, W2, Y0, Y1], dim=1)
            kernel_matrix2 = calculate_kernel_matrix(kernel_inputs2)
            
            loss2 = torch.abs(MMR_loss(pred2 * tt2, q11_pred, kernel_matrix2, loss_name='U_statistic'))
            
            loss2.backward()
            optimizer.step()
            
        # Eval
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for batch in val_loader:
                Z1, Y0, W1, tt1, Z2, Y1, W2, tt2 = [t.to(device) for t in batch]
                q11_pred = model_q11(torch.cat([Z1, Y0], dim=1))
                pred2 = model(torch.cat([Z1, Z2, Y0, Y1], dim=1))
                kernel_matrix2 = calculate_kernel_matrix(torch.cat([W1, W2, Y0, Y1], dim=1))
                
                v_loss = torch.abs(MMR_loss(pred2 * tt2, q11_pred, kernel_matrix2, loss_name='U_statistic'))
                total_val_loss += v_loss.item()
            total_val_loss /= len(val_loader)
            
        early_stopping(total_val_loss)
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_model_state = model.state_dict()
        if early_stopping.early_stop:
            break
            
    model.load_state_dict(best_model_state)
    return model, best_val_loss


def optimize_hyperparams(df_train: pd.DataFrame, df_val: pd.DataFrame, a1: int, a2: int, n_trials: int = 10, batch_size: int = 128) -> Dict[str, Any]:
    """
    使用 Optuna 进行两阶段网络架构及超参数自动调优。
    优化目标是让验证集上的 q22 Loss 达到最小。
    """
    
    train_dataset = prepare_tensors(df_train, a1, a2)
    val_dataset = prepare_tensors(df_val, a1, a2)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    def objective(trial):
        # Hyperparameters search space
        params = {
            'network_width': trial.suggest_categorical('network_width', [32, 64, 128]),
            'network_depth': trial.suggest_int('network_depth', 2, 4),
            'dropout_prob': trial.suggest_float('dropout_prob', 0.1, 0.5),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'l2': trial.suggest_float('l2', 1e-6, 1e-2, log=True),
            'epochs': 100
        }
        
        # 1. Train q11
        model_q11, _ = train_q11(train_loader, val_loader, params)
        
        # 2. Train q22
        _, val_loss_q22 = train_q22(train_loader, val_loader, model_q11, params)
        
        # 为了防止在极其糟糕的网络参数下发散，返回极大的 loss
        if np.isnan(val_loss_q22) or np.isinf(val_loss_q22):
            return 1e6
            
        return val_loss_q22

    # Run Optuna Study
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_params['epochs'] = 200 # 找到最佳结构后再稍微多跑点epoch进行最终收敛
    return best_params

def estimate_nuisance(df_train: pd.DataFrame, df_val: pd.DataFrame, a1: int, a2: int, n_trials: int = 15) -> Tuple[Callable[[pd.DataFrame], np.ndarray], nn.Module, Dict[str, Any]]:
    """
    封装了调优+最终训练的全工作流。
    返回训练好的 q22 预测器函数。
    """
    print(f"Starting Hyperparameter Optimization for a1={a1}, a2={a2}...")
    best_params = optimize_hyperparams(df_train, df_val, a1, a2, n_trials=n_trials)
    print(f"Best Params found: {best_params}")
    
    train_dataset = prepare_tensors(df_train, a1, a2)
    val_dataset = prepare_tensors(df_val, a1, a2)
    
    # 增加Batch Size加速训练
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    print("Training Final q11 Checkpoint...")
    model_q11, _ = train_q11(train_loader, val_loader, best_params)
    
    print("Training Final q22 Checkpoint...")
    model_q22, _ = train_q22(train_loader, val_loader, model_q11, best_params)
    
    # 返回一个预测闭包（可以直接作用在 df 或 tensors 上）
    def predict_q22(df_test):
        model_q22.eval()
        Z1 = torch.tensor(df_test['Z1'].values, dtype=torch.float32).unsqueeze(1).to(device)
        Z2 = torch.tensor(df_test['Z2'].values, dtype=torch.float32).unsqueeze(1).to(device)
        Y0 = torch.tensor(df_test['Y0'].values, dtype=torch.float32).unsqueeze(1).to(device)
        Y1 = torch.tensor(df_test['Y1'].values, dtype=torch.float32).unsqueeze(1).to(device)
        
        with torch.no_grad():
            preds = model_q22(torch.cat([Z1, Z2, Y0, Y1], dim=1))
        return preds.cpu().numpy().flatten()
        
    return predict_q22, model_q22, best_params
    

if __name__ == "__main__":
    print(device)