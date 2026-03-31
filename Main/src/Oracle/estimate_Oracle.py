import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import argparse
import torch
from torch.utils.data import DataLoader
import os

from Main.src.data_generate import data_gen, adjust_para_set_for_new_coding, origin_para_set
from Main.src.Oracle.step1_nuisance import estimate_nuisance
from Main.src.Oracle.step2_inner import inner_optimization
from Main.src.Oracle.step3_outer import optimize_outer_hyperparams, train_outer_policies, prepare_outer_tensors

def save_trained_models(f1, f2, best_params, n_train, tau, phi_type, model_type, seed):
    print("----Saving Oracle Models---- ")
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
    os.makedirs(models_dir, exist_ok=True)
    config_str = f"Oracle_ntrain{n_train}_tau{tau}_phi{phi_type}_model{model_type}_seed{seed}"
    torch.save({'state_dict': f1.state_dict(), 'hyperparams': best_params}, os.path.join(models_dir, f"f1_{config_str}.pt"))
    torch.save({'state_dict': f2.state_dict(), 'hyperparams': best_params}, os.path.join(models_dir, f"f2_{config_str}.pt"))
    print(f"📁 Oracle Policy Models saved with prefix: {config_str}")

def train_policy_Oracle(n_train=2000, seed=42, K_folds=2, max_alt_iters=10, tau=0.5, phi_type=1, model_type="linear", save_models=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
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

    print("\n=== Oracle Step 2 & 3: Alternating Optimization ===")
    d1_pred, d2_pred = np.ones(n_train), np.ones(n_train)
    q_current = inner_optimization(df_train['Y2'], df_train['A1'], df_train['A2'], d1_pred, d2_pred, ipw_train_oof, tau=tau)
    
    f1, f2 = None, None
    for it in range(max_alt_iters):
        print(f"--- Iter {it+1}/{max_alt_iters} (q={q_current:.6f}) ---")
        best_p = optimize_outer_hyperparams(df_train, ipw_train_oof, df_val, ipw_val_preds, q_current, n_trials=10, phi_type=phi_type, model_type=model_type)
        ds_t = prepare_outer_tensors(df_train, ipw_train_oof, q_current)
        ds_v = prepare_outer_tensors(df_val, ipw_val_preds, q_current)
        f1, f2, val_loss = train_outer_policies(DataLoader(ds_t, 128, True), DataLoader(ds_v, 128, False), best_p, phi_type, model_type)
        f1.eval()
        f2.eval()
        with torch.no_grad():
            H1 = torch.tensor(df_train['Y0'].values, dtype=torch.float32).unsqueeze(1).to(next(f1.parameters()).device)
            H2 = torch.cat([torch.tensor(df_train['Y0'].values, dtype=torch.float32).unsqueeze(1),
                            torch.tensor(df_train['Y1'].values, dtype=torch.float32).unsqueeze(1),
                            torch.tensor(df_train['A1'].values, dtype=torch.float32).unsqueeze(1)], dim=1).to(next(f2.parameters()).device)
            d1_new = np.sign(f1(H1).cpu().numpy().flatten())
            d1_new[d1_new==0]=1
            d2_new = np.sign(f2(H2).cpu().numpy().flatten())
            d2_new[d2_new==0]=1
            
            diff_ratio = 0.5 * (np.mean(d1_new != d1_pred) + np.mean(d2_new != d2_pred))
            print(f"    -> Policy Action change ratio: {diff_ratio:.6f}")
            
            d1_pred = d1_new
            d2_pred = d2_new
            
        new_q = inner_optimization(df_train['Y2'], df_train['A1'], df_train['A2'], d1_pred, d2_pred, ipw_train_oof, tau=tau)
        if it > 0 and (abs(new_q - q_current) < 1e-6) and (diff_ratio < 1e-4): 
            break
        q_current = new_q
        
    if save_models: 
        save_trained_models(f1, f2, best_p, n_train, tau, phi_type, model_type, seed)

    return f1, f2, q_current, -val_loss
