import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import argparse
import torch
from torch.utils.data import DataLoader
import scipy.stats as stats


from Main.src.prox_qtr_sl.step1_nuisance import estimate_nuisance, prepare_tensors
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

def train_policy_prox_qtr_sl(n_train=1000, seed=20026, K_folds=2, max_alt_iters=30, tau=0.5, phi_type=1, model_type="linear", save_models=False, dgp="S2"):
    """
    运行基于 Proximal QTR (Sequential Classification) 的两阶段策略学习全流程。
    支持自动切换 S1 (data_generate) / S2 (data_generate_new) 数据集来源。
    """
    
    if dgp == "S1":
        from Main.src.data_generate import data_gen, adjust_para_set_for_new_coding, origin_para_set
    else:
        from Main.src.data_generate_new import data_gen, adjust_para_set_for_new_coding, origin_para_set

    # 为了复现稳定性，统一设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("\n" + "="*50)
    print(f"🚀 Starting Proximal QTR Policy Learning ({dgp})")
    
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
                # n_trials 统一设置为 10
                predict_q22_fn, _, _ = estimate_nuisance(sub_train_fold, sub_val_fold, a1, a2, n_trials=10)
                
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


    # === 2.5 Weight Trimming (Option A: 1%/99% Trimming) ===
    def trim_weights(w, lower_p=1, upper_p=99):
        if len(w) > 0 and np.std(w) > 1e-6:
            low = np.percentile(w, lower_p)
            high = np.percentile(w, upper_p)
            return np.clip(w, low, high)
        return w

    print(f"Trimming q22 weights (1%/99% percentile)... Train mean: {np.mean(q22_train_oof):.4f}, Max: {np.max(q22_train_oof):.4f}")
    q22_train_oof = trim_weights(q22_train_oof)
    q22_val_preds = trim_weights(q22_val_preds)
    print(f"Post-trimming -> Train mean: {np.mean(q22_train_oof):.4f}, Max: {np.max(q22_train_oof):.4f}")

    # === 3. 第二/三步: 内外层交替优化 (Sequential Classification Learning) ===
    print("\n=== Step 2 & 3: Alternating Optimization for Policy Learning ===")
    
    Y2_array = df_train['Y2'].values
    A1_array = df_train['A1'].values
    A2_array = df_train['A2'].values
    
    l_bound = np.min(Y2_array)
    u_bound = np.max(Y2_array)
    epsilon_n = 0.5 / np.sqrt(n_train)
    kappa_n = np.std(Y2_array) / (6.0 * np.sqrt(n_train))
    hn = 0.2 / np.log(n_train)
    
    print(f"SCL Settings -> Initial bounds: [{l_bound:.6f}, {u_bound:.6f}], epsilon_n: {epsilon_n:.6f}, kappa_n: {kappa_n:.6f}, hn: {hn:.6f}")
    
    q_current = None
    f1, f2 = None, None
    best_sv = 0.0
    
    # 提前将验证不需要更新的张量放入目标设备，避免后续 SCL 迭代中几十次重复拷贝
    device_compute = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    H1_train_tensor = torch.tensor(df_train['Y0'].values, dtype=torch.float32).unsqueeze(1).to(device_compute)
    H2_train_tensor = torch.cat([
        torch.tensor(df_train['Y0'].values, dtype=torch.float32).unsqueeze(1),
        torch.tensor(df_train['Y1'].values, dtype=torch.float32).unsqueeze(1),
        torch.tensor(df_train['A1'].values, dtype=torch.float32).unsqueeze(1)
    ], dim=1).to(device_compute)
    
    for it in range(max_alt_iters):
        print(f"\n--- SCL Binary Search Iteration {it+1}/{max_alt_iters} ---")
        q_current = (l_bound + u_bound) / 2.0
        print(f"Testing binary boundary q_current (m) = {q_current:.6f} with bounds [{l_bound:.6f}, {u_bound:.6f}]")
        
        # Outer Level
        best_params = optimize_outer_hyperparams(df_train, q22_train_oof, df_val, q22_val_preds, 
                                                 q_current, n_trials=10, epochs=200, phi_type=phi_type, model_type=model_type)
        
        print(f"Optimal configs: {best_params}")
        
        train_dataset = prepare_outer_tensors(df_train, q22_train_oof, q_current)
        val_dataset = prepare_outer_tensors(df_val, q22_val_preds, q_current)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        
        f1, f2, val_loss = train_outer_policies(train_loader, val_loader, best_params, phi_type=phi_type, model_type=model_type)
        
        f1.eval()
        f2.eval()
        with torch.no_grad():
            f1_train_out = f1(H1_train_tensor).cpu().numpy().flatten()
            f2_train_out = f2(H2_train_tensor).cpu().numpy().flatten()
            
        phi1 = stats.norm.cdf((A1_array * f1_train_out) / hn)
        phi2 = stats.norm.cdf((A2_array * f2_train_out) / hn)
        
        sv_val = np.mean((Y2_array > q_current) * q22_train_oof * phi1 * phi2)
        best_sv = sv_val
        print(f"    -> Empirical Survival Value (SV) at {q_current:.6f}: {sv_val:.6f} (Target: {1-tau:.6f})")
        
        if abs(sv_val - (1 - tau)) <= epsilon_n:
            print(f"✅ SCL Converged by epsilon: |{sv_val:.6f} - {1-tau:.6f}| <= {epsilon_n:.6f}")
            break
        elif sv_val >= (1 - tau):
            l_bound = q_current
        else:
            u_bound = q_current
            
        if (u_bound - l_bound) <= kappa_n:
            print(f"✅ SCL Converged by kappa bounded interval width: {u_bound - l_bound:.6f} <= {kappa_n:.6f}")
            break
        
    print(f"\n✅ Training Completed! Final optimal q: {q_current:.6f}, Estimated SV: {best_sv:.6f}")
    
    if save_models: # default: False
        save_trained_models(f1, f2, best_params, n_train, tau, phi_type, model_type, seed, df_train)
    
    return f1, f2, q_current, best_sv
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proximal Quantile-Optimal DTR Pipeline")
    parser.add_argument("--n_train", type=int, default=1000, help="Number of simulation training samples")
    parser.add_argument("--seed", type=int, default=20026, help="Random seed setting for reproducibility")
    parser.add_argument("--tau", type=float, default=0.5, help="Quantile level (e.g. 0.5 for median)")
    parser.add_argument("--folds", type=int, default=2, help="K-folds config for Cross-fitting in Nuisance pre-estimation")
    parser.add_argument("--phi_type", type=int, default=1, choices=[1, 2, 3, 4], help="Type of surrogate loss phi(x) (1 to 4)")
    parser.add_argument("--model_type", type=str, default="linear", choices=["linear", "nn"], help="Type of policy network (linear or nn)")
    args = parser.parse_args()
    
    train_policy_prox_qtr_sl(args.n_train, args.seed, args.folds, tau=args.tau, phi_type=args.phi_type, model_type=args.model_type, save_models=True)
