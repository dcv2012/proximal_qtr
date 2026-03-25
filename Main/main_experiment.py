import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import argparse
import torch

from Main.src.data_generate import data_gen, adjust_para_set_for_new_coding, origin_para_set
from Main.src.step1_nuisance import estimate_nuisance, prepare_tensors
from Main.src.step2_inner import inner_optimization
from Main.src.step3_outer import optimize_outer_hyperparams, train_outer_policies, prepare_outer_tensors

def run_experiment(n_train=2000, n_val=500, n_test=1000, seed=42, K_folds=2, max_alt_iters=3, tau=0.5, phi_type=1, model_type="linear"):
    # 设定随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # === 1. 数据生成 ===
    print("=== Configuration & Data Generation ===")
    params = adjust_para_set_for_new_coding(origin_para_set)
    df_train = data_gen(n_train, params)
    df_val = data_gen(n_val, params)
    df_test = data_gen(n_test, params)
    print(f"Generated data: Train({n_train}), Val({n_val}), Test({n_test})")
    
    
    # === 2. 第一步: 连续估计滋扰（桥函数 q22）及交叉拟合 ===
    print("\n=== Step 1: Pre-estimating Bridge Functions (q22) w/ Cross-Fitting ===")
    
    q22_train_oof = np.zeros(len(df_train))
    q22_val_preds = np.zeros(len(df_val))
    
    kf = KFold(n_splits=K_folds, shuffle=True, random_state=seed)
    
    for fold, (train_idx, oof_idx) in enumerate(kf.split(df_train)):
        print(f"\n>> Cross-Fitting Fold {fold+1}/{K_folds}")
        df_train_fold = df_train.iloc[train_idx].reset_index(drop=True)
        df_oof_fold = df_train.iloc[oof_idx].reset_index(drop=True)
        
        # 针对每个治疗方案对建立模型
        for a1 in [1, -1]:
            for a2 in [1, -1]:
                # 如果当前fold完全没有针对这种方案的数据，则跳过
                if not ((df_train_fold['A1'] == a1) & (df_train_fold['A2'] == a2)).any():
                    continue
                    
                # The validation fold acts as early-stopping validation target for hyperparameter tuning.
                # n_trials 设较小以提升模拟速度，生产环境中可加大至例如 20-50
                predict_q22_fn = estimate_nuisance(df_train_fold, df_oof_fold, a1, a2, n_trials=10)
                
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
    for it in range(max_alt_iters):
        print(f"\n--- Alternating Iteration {it+1}/{max_alt_iters} ---")
        print(f"Fixing q = {q_current:.4f}, optimizing outer policies f1, f2 with Optuna...")
        
        # Outer Level
        best_params = optimize_outer_hyperparams(df_train, q22_train_oof, df_val, q22_val_preds, 
                                                 q_current, n_trials=5, epochs=100, phi_type=phi_type, model_type=model_type)
        
        print(f"Optimal configs: {best_params}")
        from torch.utils.data import DataLoader
        train_dataset = prepare_outer_tensors(df_train, q22_train_oof, q_current)
        val_dataset = prepare_outer_tensors(df_val, q22_val_preds, q_current)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        f1, f2, _ = train_outer_policies(train_loader, val_loader, best_params, phi_type=phi_type, model_type=model_type)
        
        f1.eval(); f2.eval()
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
        
    print("\n✅ Experiment logic built effectively! Pipeline is ready to process more combinations.")
    return f1, f2, q_current


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proximal Quantile-Optimal DTR Pipeline")
    parser.add_argument("--n_train", type=int, default=1000, help="Number of simulation training samples")
    parser.add_argument("--n_val", type=int, default=500, help="Number of simulated validation samples")
    parser.add_argument("--n_test", type=int, default=1000, help="Number of simulated testing samples")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed setting for reproducibility")
    parser.add_argument("--tau", type=float, default=0.5, help="Quantile level (e.g. 0.5 for median)")
    parser.add_argument("--folds", type=int, default=2, help="K-folds config for Cross-fitting in Nuisance pre-estimation")
    parser.add_argument("--phi_type", type=int, default=1, choices=[1, 2, 3, 4], help="Type of surrogate loss phi(x) (1 to 4)")
    parser.add_argument("--model_type", type=str, default="linear", choices=["linear", "nn"], help="Type of policy network (linear or nn)")
    args = parser.parse_args()
    
    run_experiment(args.n_train, args.n_val, args.n_test, args.seed, args.folds, tau=args.tau, phi_type=args.phi_type, model_type=args.model_type)
