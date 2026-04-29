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


def extract_proxy(df: pd.DataFrame, prefix: str) -> torch.Tensor:
    """自动适配一维或多维代理变量的新旧DGP"""
    if prefix in df.columns:
        return torch.tensor(df[prefix].values, dtype=torch.float32).unsqueeze(1)
    cols = [c for c in df.columns if c.startswith(prefix)]
    cols.sort()
    return torch.tensor(df[cols].values, dtype=torch.float32)

def prepare_tensors(df: pd.DataFrame, a1: int, a2: int) -> TensorDataset:
    """
    使用全量数据构建张量集，利用 tt1 和 tt2 来过滤或惩罚样本，保留因果调整中不可或缺的协变量基础分布。
    """
    Y0 = torch.tensor(df['Y0'].values, dtype=torch.float32).unsqueeze(1)
    Y1 = torch.tensor(df['Y1'].values, dtype=torch.float32).unsqueeze(1)
    A1 = torch.tensor(df['A1'].values, dtype=torch.float32).unsqueeze(1)
    
    # 动态支持 Z1, W1, Z2, W2 的高维扩展
    Z1 = extract_proxy(df, 'Z1')
    W1 = extract_proxy(df, 'W1')
    Z2 = extract_proxy(df, 'Z2')
    W2 = extract_proxy(df, 'W2')
    
    # tt1: I(A1=a1) 全分布指示掩码
    tt1 = torch.tensor((df['A1'] == a1).values, dtype=torch.float32).unsqueeze(1)
    
    # tt2: I(A1=a1 and A2=a2) 序贯时序掩码
    tt2 = torch.tensor(((df['A1'] == a1) & (df['A2'] == a2)).values, dtype=torch.float32).unsqueeze(1)
    
    return TensorDataset(Z1, Y0, A1, W1, tt1, Z2, Y1, W2, tt2)


def train_q11(train_loader: DataLoader, val_loader: DataLoader, params: Dict[str, Any], mmr_loss_type: str = 'V_statistic') -> Tuple[nn.Module, float]:
    # 动态抓取维度：Z1_dim + Y0_dim
    Z1_peek, Y0_peek, _, _, _, _, _, _, _ = next(iter(train_loader))
    input_dim = Z1_peek.shape[1] + Y0_peek.shape[1]
    
    model = MLP_for_MMR(input_dim, params).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['l2'])
    
    early_stopping = EarlyStopping()
    best_model_state = model.state_dict() # 初始状态占位，防止训练全量发散时返回 None
    best_val_loss = float('inf')
    
    for epoch in range(params['epochs']):
        model.train()
        for batch in train_loader:
            Z1, Y0, _, W1, tt1, _, _, _, _ = [t.to(device) for t in batch]
            
            optimizer.zero_grad()
            pred = model(torch.cat([Z1, Y0], dim=1))
            
            kernel_inputs = torch.cat([W1, Y0], dim=1)
            kernel_matrix = calculate_kernel_matrix(kernel_inputs)
            
            loss = torch.abs(MMR_loss(pred * tt1, torch.ones_like(pred), kernel_matrix, loss_name=mmr_loss_type, lambda_reg=params.get('lambda_reg', 0.0)))
            loss.backward()
            optimizer.step()
            
        # Eval
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for batch in val_loader:
                Z1, Y0, _, W1, tt1, _, _, _, _ = [t.to(device) for t in batch]
                pred = model(torch.cat([Z1, Y0], dim=1))
                kernel_matrix = calculate_kernel_matrix(torch.cat([W1, Y0], dim=1))
                v_loss = torch.abs(MMR_loss(pred * tt1, torch.ones_like(pred), kernel_matrix, loss_name=mmr_loss_type, lambda_reg=params.get('lambda_reg', 0.0)))
                total_val_loss += v_loss.item()
            total_val_loss /= len(val_loader)
            
        early_stopping(total_val_loss)
        # 增加对 NaN 的健壮性判断
        if not np.isnan(total_val_loss) and total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_model_state = model.state_dict()
        if early_stopping.early_stop:
            break
            
    model.load_state_dict(best_model_state)
    return model, best_val_loss

def train_q22(train_loader: DataLoader, val_loader: DataLoader, model_q11: nn.Module, params: Dict[str, Any], mmr_loss_type: str = 'V_statistic') -> Tuple[nn.Module, float]:
    # 动态抓取维度：Z1_dim + Z2_dim + Y0_dim + Y1_dim
    Z1_peek, Y0_peek, A1_peek, _, _, Z2_peek, Y1_peek, _, _ = next(iter(train_loader))
    input_dim = Z1_peek.shape[1] + Z2_peek.shape[1] + Y0_peek.shape[1] + Y1_peek.shape[1] + A1_peek.shape[1]
    
    q22_params = params.copy()
    q22_params['output_bound'] = params.get('q22_output_bound', 5.0)
    
    model = MLP_for_MMR(input_dim, q22_params).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['l2'])
    
    early_stopping = EarlyStopping()
    best_model_state = model.state_dict() # 初始状态占位
    best_val_loss = float('inf')
    
    model_q11.eval()
    
    for epoch in range(params['epochs']):
        model.train()
        for batch in train_loader:
            Z1, Y0, A1, W1, tt1, Z2, Y1, W2, tt2 = [t.to(device) for t in batch]
            
            optimizer.zero_grad()
            
            # Predict q11 (Cached dynamically per batch, no gradient)
            with torch.no_grad():
                q11_pred = model_q11(torch.cat([Z1, Y0], dim=1))
            
            # Predict q22
            pred2 = model(torch.cat([Z1, Z2, Y0, Y1, A1], dim=1))
            kernel_inputs2 = torch.cat([W1, W2, Y0, Y1, A1], dim=1)
            kernel_matrix2 = calculate_kernel_matrix(kernel_inputs2)
            
            q11_target = q11_pred * tt1
            loss2 = torch.abs(MMR_loss(pred2 * tt2, q11_target, kernel_matrix2, loss_name=mmr_loss_type, lambda_reg=params.get('lambda_reg', 0.0)))
            
            loss2.backward()
            optimizer.step()
            
        # Eval
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for batch in val_loader:
                Z1, Y0, A1, W1, tt1, Z2, Y1, W2, tt2 = [t.to(device) for t in batch]
                q11_pred = model_q11(torch.cat([Z1, Y0], dim=1))
                pred2 = model(torch.cat([Z1, Z2, Y0, Y1, A1], dim=1))
                kernel_matrix2 = calculate_kernel_matrix(torch.cat([W1, W2, Y0, Y1, A1], dim=1))
                
                q11_target = q11_pred * tt1
                v_loss = torch.abs(MMR_loss(pred2 * tt2, q11_target, kernel_matrix2, loss_name=mmr_loss_type, lambda_reg=params.get('lambda_reg', 0.0)))
                total_val_loss += v_loss.item()
            total_val_loss /= len(val_loader)
            
        early_stopping(total_val_loss)
        if not np.isnan(total_val_loss) and total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_model_state = model.state_dict()
        if early_stopping.early_stop:
            break
            
    model.load_state_dict(best_model_state)
    return model, best_val_loss


def optimize_hyperparams(df_train: pd.DataFrame, df_val: pd.DataFrame, a1: int, a2: int, n_trials: int = 10, batch_size: int = 128, mmr_loss_type: str = 'V_statistic', q22_output_bound: float = 5.0) -> Dict[str, Any]:
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
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            'l2': trial.suggest_float('l2', 1e-4, 1e-2, log=True),
            'lambda_reg': trial.suggest_float('lambda_reg', 1e-6, 1e-2, log=True),
            'q22_output_bound': q22_output_bound,
            'epochs': 200
        }
        
        # 1. Train q11
        model_q11, _ = train_q11(train_loader, val_loader, params, mmr_loss_type=mmr_loss_type)
        
        # 2. Train q22
        _, val_loss_q22 = train_q22(train_loader, val_loader, model_q11, params, mmr_loss_type=mmr_loss_type)
        
        # 为了防止在极其糟糕的网络参数下发散，返回极大的 loss
        if np.isnan(val_loss_q22) or np.isinf(val_loss_q22):
            return 1e6
            
        return val_loss_q22

    # Run Optuna Study
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=torch.randint(0, 100000, (1,)).item()))
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_params['q22_output_bound'] = q22_output_bound
    best_params['epochs'] = 200 # 找到最佳结构后再稍微多跑点epoch进行最终收敛
    return best_params

def estimate_nuisance(df_train: pd.DataFrame, df_val: pd.DataFrame, a1: int, a2: int, n_trials: int = 10, mmr_loss_type: str = 'V_statistic', q22_output_bound: float = 5.0) -> Tuple[Callable[[pd.DataFrame], np.ndarray], nn.Module, Dict[str, Any]]:
    """
    封装了调优+最终训练的全工作流。
    返回训练好的 q22 预测器函数。
    """
    print(f"Starting Hyperparameter Optimization for a1={a1}, a2={a2}...")
    best_params = optimize_hyperparams(df_train, df_val, a1, a2, n_trials=n_trials, mmr_loss_type=mmr_loss_type, q22_output_bound=q22_output_bound)
    print(f"Best Params found: {best_params}")

    train_dataset = prepare_tensors(df_train, a1, a2)
    val_dataset = prepare_tensors(df_val, a1, a2)
    
    # 增加Batch Size加速训练
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    print("Training Final q11 Checkpoint...")
    model_q11, val_q11 = train_q11(train_loader, val_loader, best_params, mmr_loss_type=mmr_loss_type)
    if val_q11 == float('inf'):
        print(f"Warning: Final q11 model diverged for A1={a1}, A2={a2}. Continuing with fallback initial weights.")
    
    print("Training Final q22 Checkpoint...")
    model_q22, val_q22 = train_q22(train_loader, val_loader, model_q11, best_params, mmr_loss_type=mmr_loss_type)
    if val_q22 == float('inf'):
        print(f"Warning: Final q22 model diverged for A1={a1}, A2={a2}. Continuing with fallback initial weights.")
    
    # 返回一个预测闭包（可以直接作用在 df 或 tensors 上）
    def predict_q22(df_test):
        model_q22.eval()
        Z1 = extract_proxy(df_test, 'Z1').to(device)
        Z2 = extract_proxy(df_test, 'Z2').to(device)
        Y0 = torch.tensor(df_test['Y0'].values, dtype=torch.float32).unsqueeze(1).to(device)
        Y1 = torch.tensor(df_test['Y1'].values, dtype=torch.float32).unsqueeze(1).to(device)
        A1 = torch.tensor(df_test['A1'].values, dtype=torch.float32).unsqueeze(1).to(device)
        
        with torch.no_grad():
            preds = model_q22(torch.cat([Z1, Z2, Y0, Y1, A1], dim=1))
            
        preds_np = preds.cpu().numpy().flatten()
        
        if not np.isfinite(preds_np).all():
            print(f"Warning: predict_q22 produced NaN or Inf for combo A1={a1}, A2={a2}. Auto-correcting.")
            # 将 NaN 替换为 0.0 (无实际贡献)，将 Inf 截断在一定范围内
            preds_np = np.nan_to_num(preds_np, nan=0.0, posinf=1e4, neginf=-1e4)
        
        return preds_np
        
    return predict_q22, model_q22, best_params
    

if __name__ == "__main__":
    print(device)
