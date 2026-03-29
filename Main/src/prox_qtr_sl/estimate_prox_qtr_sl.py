import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import argparse
import torch
from torch.utils.data import DataLoader

from Main.src.data_generate import data_gen, adjust_para_set_for_new_coding, origin_para_set
from Main.src.prox_qtr_sl.step1_nuisance import estimate_nuisance, prepare_tensors
from Main.src.prox_qtr_sl.step2_inner import inner_optimization
from Main.src.prox_qtr_sl.step3_outer import optimize_outer_hyperparams, train_outer_policies, prepare_outer_tensors

import os

def save_trained_models(f1, f2, best_params, n_train, tau, phi_type, model_type, seed, df_train):
    
    print("----Saving Models---- ")
    # 模型统一保存在 Main/models 目录下 (当前文件在 Main/src/prox_qtr_sl/)
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
    os.makedirs(models_dir, exist_ok=True)
    
    config_str = f"ntrain{n_train}_tau{tau}_phi{phi_type}_model{model_type}_seed{seed}"
    f1_path = os.path.join(models_dir, f"f1_{config_str}.pt")
    f2_path = os.path.join(models_dir, f"f2_{config_str}.pt")
    
    # 因为存在隐层和结构超参数的动态变动，打包存储 params
    torch.save({'state_dict': f1.state_dict(), 'hyperparams': best_params}, f1_path)
    torch.save({'state_dict': f2.state_dict(), 'hyperparams': best_params}, f2_path)
    print(f"📁 Policy Models gracefully saved to:\n  - {f1_path}\n  - {f2_path}")
    
    print("\n=== Post-Training: Training Full-Sample q22 Models for Offline Evaluation ===")
    
    sub_train_full, sub_val_full = train_test_split(df_train, test_size=0.2, random_state=seed)
    
    for a1 in [1, -1]:
        for a2 in [1, -1]:
            if not ((sub_train_full['A1'] == a1) & (sub_train_full['A2'] == a2)).any():
                continue
            print(f"-> Assuring final q22 model for A1={a1}, A2={a2} on full training data...")
            _, final_q22_model, final_params = estimate_nuisance(sub_train_full, sub_val_full, a1, a2, n_trials=3)
            
            q22_path = os.path.join(models_dir, f"q22_a1_{a1}_a2_{a2}_{n_train}.pt")
            torch.save({'state_dict': final_q22_model.state_dict(), 'hyperparams': final_params}, q22_path)
            
    print("📁 4 standalone q22 Nuisance Models securely saved for offline evaluation!")

def train_policy_prox_qtr_sl(n_train=2000, seed=42, K_folds=2, max_alt_iters=3, tau=0.5, phi_type=1, model_type="linear", save_models=False):
    # 设定随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 移除外界对于验证集大小的硬编码输入，直接绑定为训练集比例的0.25倍
    n_val = int(n_train * 0.25)
    
    # === 1. 数据生成 ===
    print("=== Configuration & Data Generation ===")
    params = adjust_para_set_for_new_coding(origin_para_set)
    df_train = data_gen(n_train, params)
    df_val = data_gen(n_val, params)
    print(f"Generated data: Train({n_train}), Val({n_val})")
    
    
    # === 2. 第一步: 连续估计滋扰（桥函数 q22）及交叉拟合 ===
    print("\n=== Step 1: Pre-estimating Bridge Functions (q22) w/ Cross-Fitting ===")
    
    q22_train_oof = np.zeros(len(df_train))
    q22_val_preds = np.zeros(len(df_val))
    
    kf = KFold(n_splits=K_folds, shuffle=True, random_state=seed)
    
    for fold, (train_idx, oof_idx) in enumerate(kf.split(df_train)):
        print(f"\n>> Cross-Fitting Fold {fold+1}/{K_folds}")
        df_train_fold = df_train.iloc[train_idx].reset_index(drop=True)
        df_oof_fold = df_train.iloc[oof_idx].reset_index(drop=True)
        
        # 警告：为了严格满足 Cross-Fitting 的无偏性(独立性)假设，被留出评估的 df_oof_fold 绝对不能参与前期的任何调参!
        # 否则通过 Optuna 和早停机制，模型会泄露关于本折目标集的信息，从而破坏双重健壮机制。
        # 正确做法：从不互斥的训练集(df_train_fold)内部自己切出完全独立于 OOF 的 20% 作为 tuning 验证集。
        sub_train_fold, sub_val_fold = train_test_split(df_train_fold, test_size=0.2, random_state=seed)
        
        # 针对每个治疗方案对建立模型
        for a1 in [1, -1]:
            for a2 in [1, -1]:
                # 如果当前fold完全没有针对这种方案的数据，则跳过
                if not ((sub_train_fold['A1'] == a1) & (sub_train_fold['A2'] == a2)).any():
                    continue
                    
                # 仅仅用内部的 sub_train 和 sub_val 联合预测，导出的预估器对于 df_oof_fold 将完全保持无偏
                # n_trials 设较小以提升模拟速度，生产环境中可加大至例如 20-50
                predict_q22_fn, _, _ = estimate_nuisance(sub_train_fold, sub_val_fold, a1, a2, n_trials=5)
                
                # 对 OOF 集合做预测，但只针对匹配 (A_1=a_1, A_2=a_2) 的观测
                oof_sub_mask = (df_oof_fold['A1'] == a1) & (df_oof_fold['A2'] == a2)
                if oof_sub_mask.sum() > 0:
                    matching_oof_data = df_oof_fold[oof_sub_mask].copy()
                    
                    # 取回原数据集级别的 index
                    global_indices = df_train.iloc[oof_idx][oof_sub_mask.values].index
                    preds_part = predict_q22_fn(matching_oof_data)
                    q22_train_oof[global_indices] += preds_part
                    
                # 对于独立的 validation 集进行普通的多模型集平均 (Ensemble)
                val_sub_mask = (df_val['A1'] == a1) & (df_val['A2'] == a2)
                if val_sub_mask.sum() > 0:
                    matching_val_df = df_val[val_sub_mask].copy()
                    global_val_idx = df_val[val_sub_mask.values].index
                    
                    preds_val_part = predict_q22_fn(matching_val_df)
                    q22_val_preds[global_val_idx] += preds_val_part / K_folds


    # === 3. 第二/三步: 内外层交替优化 ===
    print("\n=== Step 2 & 3: Alternating Optimization for Policy Learning ===")
    # 策略初始化
    d1_pred = np.ones(n_train)
    d2_pred = np.ones(n_train)
    
    # 初始 Inner Optimization -> q
    q_current = inner_optimization(df_train['Y2'], df_train['A1'], df_train['A2'], 
                                   d1_pred, d2_pred, q22_train_oof, tau=tau)
    print(f"Initial Naive q: {q_current:.4f}")
    
    f1, f2 = None, None
    best_sv = 0.0
    for it in range(max_alt_iters):
        print(f"\n--- Alternating Iteration {it+1}/{max_alt_iters} ---")
        print(f"Fixing q = {q_current:.4f}, optimizing outer policies f1, f2 with Optuna...")
        
        # Outer Level
        best_params = optimize_outer_hyperparams(df_train, q22_train_oof, df_val, q22_val_preds, 
                                                 q_current, n_trials=5, epochs=100, phi_type=phi_type, model_type=model_type)
        
        print(f"Optimal configs: {best_params}")
        
        train_dataset = prepare_outer_tensors(df_train, q22_train_oof, q_current)
        val_dataset = prepare_outer_tensors(df_val, q22_val_preds, q_current)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        f1, f2, val_loss = train_outer_policies(train_loader, val_loader, best_params, phi_type=phi_type, model_type=model_type)
        best_sv = -val_loss
        
        f1.eval()
        f2.eval()
        with torch.no_grad():
            device_f1 = next(f1.parameters()).device
            device_f2 = next(f2.parameters()).device
            H1_train = torch.tensor(df_train['Y0'].values, dtype=torch.float32).unsqueeze(1).to(device_f1)
            H2_train = torch.cat([torch.tensor(df_train['Y0'].values, dtype=torch.float32).unsqueeze(1),
                                  torch.tensor(df_train['Y1'].values, dtype=torch.float32).unsqueeze(1),
                                  torch.tensor(df_train['A1'].values, dtype=torch.float32).unsqueeze(1)], dim=1).to(device_f2)
                                  
            f1_train_out = f1(H1_train).cpu().numpy().flatten()
            f2_train_out = f2(H2_train).cpu().numpy().flatten()
            
            # 使用符号函数生成新策略指示器
            d1_pred = np.sign(f1_train_out)
            d1_pred[d1_pred == 0] = 1 # boundary decision
            
            d2_pred = np.sign(f2_train_out)
            d2_pred[d2_pred == 0] = 1
            
        # Inner Level -> Update q
        new_q = inner_optimization(df_train['Y2'], df_train['A1'], df_train['A2'], 
                                   d1_pred, d2_pred, q22_train_oof, tau=tau)
        print(f"Updated optimal q: {new_q:.4f} (Previous q: {q_current:.4f})")
        
        if np.abs(new_q - q_current) < 1e-4:
            print("Quantile constraint bounds have successfully converged!")
            q_current = new_q
            break
            
        q_current = new_q
        
    print(f"\n✅ Training Completed! Final optimal q: {q_current:.4f}, Final SV_psi: {best_sv:.4f}")
    
    if save_models: # default: False
        save_trained_models(f1, f2, best_params, n_train, tau, phi_type, model_type, seed, df_train)
    
    return f1, f2, q_current, best_sv
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proximal Quantile-Optimal DTR Pipeline")
    parser.add_argument("--n_train", type=int, default=1000, help="Number of simulation training samples")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed setting for reproducibility")
    parser.add_argument("--tau", type=float, default=0.5, help="Quantile level (e.g. 0.5 for median)")
    parser.add_argument("--folds", type=int, default=2, help="K-folds config for Cross-fitting in Nuisance pre-estimation")
    parser.add_argument("--phi_type", type=int, default=1, choices=[1, 2, 3, 4], help="Type of surrogate loss phi(x) (1 to 4)")
    parser.add_argument("--model_type", type=str, default="linear", choices=["linear", "nn"], help="Type of policy network (linear or nn)")
    args = parser.parse_args()
    
    train_policy_prox_qtr_sl(args.n_train, args.seed, args.folds, tau=args.tau, phi_type=args.phi_type, model_type=args.model_type, save_models=True)
