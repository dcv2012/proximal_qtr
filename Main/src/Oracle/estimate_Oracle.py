import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import argparse
import torch
from torch.utils.data import DataLoader
import scipy.stats as stats
import os


from Main.src.Oracle.step1_nuisance import estimate_nuisance
from Main.src.Oracle.step3_outer import optimize_outer_hyperparams, train_outer_policies, prepare_outer_tensors

def save_trained_models(f1, f2, best_params, n_train, tau, phi_type, model_type, seed):
    print("----Saving Oracle Models---- ")
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
    os.makedirs(models_dir, exist_ok=True)
    config_str = f"Oracle_ntrain{n_train}_tau{tau}_phi{phi_type}_model{model_type}_seed{seed}"
    torch.save({'state_dict': f1.state_dict(), 'hyperparams': best_params}, os.path.join(models_dir, f"f1_{config_str}.pt"))
    torch.save({'state_dict': f2.state_dict(), 'hyperparams': best_params}, os.path.join(models_dir, f"f2_{config_str}.pt"))
    print(f"📁 Oracle Policy Models saved with prefix: {config_str}")

def train_policy_Oracle(n_train=1000, seed=20026, K_folds=2, max_alt_iters=30, tau=0.5, phi_type=1, model_type="linear", save_models=False, dgp="S2"):
    """
    运行基于 Oracle 的全观察（含U）环境策略学习。
    """
    
    if dgp == "S1":
        from Main.src.data_generate import data_gen, adjust_para_set_for_new_coding, origin_para_set
    else:
        from Main.src.data_generate_new import data_gen, adjust_para_set_for_new_coding, origin_para_set

    # 为了复现稳定性，统一设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("\n" + "="*50)
    print(f"🚀 Starting Oracle Policy Learning ({dgp})")
    
    n_val = int(n_train * 0.25)
    params_dgp = adjust_para_set_for_new_coding(origin_para_set)
    df_train = data_gen(n_train, params_dgp)
    df_val = data_gen(n_val, params_dgp)
    
    print("\n=== Oracle Step 1: Estimating Propensity Weights (Logistic Regression - Oracle Features) ===")
    
    ipw_train_oof = np.zeros(len(df_train))
    ipw_val_preds = np.zeros(len(df_val))
    
    kf = KFold(n_splits=K_folds, shuffle=True, random_state=seed)
    
    for fold, (train_idx, oof_idx) in enumerate(kf.split(df_train)):
        print(f">> Fold {fold+1}/{K_folds}")
        df_t = df_train.iloc[train_idx].reset_index(drop=True)
        df_oof = df_train.iloc[oof_idx].reset_index(drop=True)
        sub_t, sub_v = train_test_split(df_t, test_size=0.2, random_state=seed)
        predict_weights_fn, _ = estimate_nuisance(sub_t, sub_v)
        ipw_train_oof[df_train.iloc[oof_idx].index] = predict_weights_fn(df_oof)
        ipw_val_preds += predict_weights_fn(df_val) / K_folds

    print("\n=== Oracle Step 2 & 3: Alternating Optimization (Sequential Classification Learning) ===")
    
    Y2_array = df_train['Y2'].values
    A1_array = df_train['A1'].values
    A2_array = df_train['A2'].values
    
    l_bound = np.min(Y2_array)
    u_bound = np.max(Y2_array)
    epsilon_n = min(1e-4, 0.5 / np.sqrt(n_train))
    kappa_n = min(1e-4, np.std(Y2_array) / (6 * np.sqrt(n_train)))
    hn = 0.2 / np.log(n_train)
    
    print(f"SCL Settings -> Initial bounds: [{l_bound:.6f}, {u_bound:.6f}], epsilon_n: {epsilon_n:.6f}, kappa_n: {kappa_n:.6f}, hn: {hn:.6f}")
    
    q_current = None
    f1, f2 = None, None
    best_sv = 0.0
    
    device_compute = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    H1_train_tensor = torch.tensor(df_train['Y0'].values, dtype=torch.float32).unsqueeze(1).to(device_compute)
    H2_train_tensor = torch.cat([
        torch.tensor(df_train['Y0'].values, dtype=torch.float32).unsqueeze(1),
        torch.tensor(df_train['Y1'].values, dtype=torch.float32).unsqueeze(1),
        torch.tensor(df_train['A1'].values, dtype=torch.float32).unsqueeze(1)
    ], dim=1).to(device_compute)
    
    for it in range(max_alt_iters):
        q_current = (l_bound + u_bound) / 2.0
        print(f"--- Iter {it+1}/{max_alt_iters} Binary Search m (q) = {q_current:.6f} with bounds [{l_bound:.6f}, {u_bound:.6f}] ---")
        best_p = optimize_outer_hyperparams(df_train, ipw_train_oof, df_val, ipw_val_preds, q_current, n_trials=10, phi_type=phi_type, model_type=model_type)
        ds_t = prepare_outer_tensors(df_train, ipw_train_oof, q_current)
        ds_v = prepare_outer_tensors(df_val, ipw_val_preds, q_current)
        f1, f2, val_loss = train_outer_policies(DataLoader(ds_t, 128, True), DataLoader(ds_v, 128, False), best_p, phi_type, model_type)
        f1.eval()
        f2.eval()
        with torch.no_grad():
            f1_train_out = f1(H1_train_tensor).cpu().numpy().flatten()
            f2_train_out = f2(H2_train_tensor).cpu().numpy().flatten()

        phi1 = stats.norm.cdf((A1_array * f1_train_out) / hn)
        phi2 = stats.norm.cdf((A2_array * f2_train_out) / hn)
            
        sv_val = np.mean((Y2_array > q_current) * ipw_train_oof * phi1 * phi2)
        best_sv = sv_val
        print(f"    -> Empirical Survival Value (SV) at {q_current:.6f}: {sv_val:.6f} (Target: {1-tau:.6f})")
        
        if abs(sv_val - (1 - tau)) <= epsilon_n:
            print(f"✅ Oracle SCL Converged by epsilon: |{sv_val:.6f} - {1-tau:.6f}| <= {epsilon_n:.6f}")
            break
        elif sv_val >= (1 - tau):
            l_bound = q_current
        else:
            u_bound = q_current
            
        if (u_bound - l_bound) <= kappa_n:
            print(f"✅ Oracle SCL Converged by kappa width: {u_bound - l_bound:.6f} <= {kappa_n:.6f}")
            break

    if save_models: 
        save_trained_models(f1, f2, best_p, n_train, tau, phi_type, model_type, seed)

    return f1, f2, q_current, best_sv
