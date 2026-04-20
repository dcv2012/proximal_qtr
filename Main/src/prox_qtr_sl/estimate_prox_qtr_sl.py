import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import argparse
import torch
from torch.utils.data import DataLoader
import scipy.stats as stats


from Main.src.prox_qtr_sl.step1_nuisance import estimate_nuisance, prepare_tensors
from Main.src.prox_qtr_sl.step3_outer import optimize_outer_hyperparams, train_outer_policies, prepare_outer_tensors
from Main.src.prox_qtr_sl.step2_inner import inner_optimization_grid

import os


def make_treatment_strata(df):
    return df['A1'].astype(str) + "_" + df['A2'].astype(str)


def split_fold_with_combo_support(df_fold, test_size=0.2, seed=0):
    """
    Create an internal tuning split while guaranteeing that every observed
    treatment combination in df_fold remains represented in the training split.
    """
    rng = np.random.RandomState(seed)
    val_indices = []

    for _, combo_df in df_fold.groupby(['A1', 'A2'], sort=False):
        combo_indices = combo_df.index.to_numpy().copy()
        rng.shuffle(combo_indices)

        if len(combo_indices) <= 1:
            continue

        n_val = int(round(len(combo_indices) * test_size))
        n_val = max(1, n_val)
        n_val = min(n_val, len(combo_indices) - 1)
        val_indices.extend(combo_indices[:n_val].tolist())

    if len(val_indices) == 0 or len(val_indices) == len(df_fold):
        raise ValueError(
            "Unable to create an internal tuning split with treatment-combination support. "
            "Increase n_train or reduce K_folds."
        )

    val_index_set = set(val_indices)
    train_indices = [idx for idx in df_fold.index if idx not in val_index_set]

    sub_train_fold = df_fold.loc[train_indices].reset_index(drop=True)
    sub_val_fold = df_fold.loc[val_indices].reset_index(drop=True)
    return sub_train_fold, sub_val_fold

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
    elif dgp == "S2":
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
    q22_train_oof_counts = np.zeros(len(df_train), dtype=int)
    q22_val_preds = np.zeros(len(df_val))
    q22_val_pred_counts = np.zeros(len(df_val), dtype=int)

    train_strata = make_treatment_strata(df_train)
    combo_counts = train_strata.value_counts().sort_index()
    min_combo_count = int(combo_counts.min())
    if min_combo_count < K_folds:
        raise ValueError(
            f"K_folds={K_folds} is too large for cross-fitting: the smallest observed "
            f"(A1, A2) cell has only {min_combo_count} samples. Reduce K_folds or increase n_train."
        )
    
    # 方案A：强制要求切分后 df_train_fold 至少有2个样本，以保证 sub_val_fold 能分到数据
    required_min = int(np.ceil(2 * K_folds / (K_folds - 1))) if K_folds > 1 else 2
    if min_combo_count < required_min:
        raise ValueError(
            f"Cross-fitting needs at least {required_min} samples for the rarest combo to support internal splits (currently {min_combo_count})."
        )

    kf = StratifiedKFold(n_splits=K_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, oof_idx) in enumerate(kf.split(df_train, train_strata)):
        print(f"\n>> Cross-Fitting Fold {fold+1}/{K_folds}")
        df_train_fold = df_train.iloc[train_idx].reset_index(drop=True)
        df_oof_fold = df_train.iloc[oof_idx].reset_index(drop=True)
        
        # 警告：为了严格满足 Cross-Fitting 的无偏性(独立性)假设，被留出评估的 df_oof_fold 绝对不能参与前期的任何调参!
        # 否则通过 Optuna 和早停机制，模型会泄露关于本折目标集的信息，从而破坏双重健壮机制。
        # 正确做法：从不互斥的训练集(df_train_fold)内部自己切出完全独立于 OOF 的 20% 作为 tuning 验证集。
        sub_train_fold, sub_val_fold = split_fold_with_combo_support(
            df_train_fold, test_size=0.2, seed=seed + fold
        )
        
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
                    q22_train_oof[global_indices] = preds_part
                    q22_train_oof_counts[global_indices] += 1
                    
                # 对于独立的 validation 集进行普通的多模型集平均 (Ensemble)
                val_sub_mask = (df_val['A1'] == a1) & (df_val['A2'] == a2)
                if val_sub_mask.sum() > 0:
                    matching_val_df = df_val[val_sub_mask].copy()
                    global_val_idx = df_val[val_sub_mask.values].index
                    
                    preds_val_part = predict_q22_fn(matching_val_df)
                    q22_val_preds[global_val_idx] += preds_val_part
                    q22_val_pred_counts[global_val_idx] += 1

    uncovered_train_mask = q22_train_oof_counts != 1
    if np.any(uncovered_train_mask):
        uncovered_summary = (
            df_train.loc[uncovered_train_mask, ['A1', 'A2']]
            .value_counts()
            .sort_index()
            .to_dict()
        )
        raise ValueError(
            "Cross-fitting failed to produce exactly one OOF q22 prediction per training sample. "
            f"Uncovered or duplicated cells: {uncovered_summary}"
        )

    uncovered_val_mask = q22_val_pred_counts == 0
    if np.any(uncovered_val_mask):
        uncovered_val_summary = (
            df_val.loc[uncovered_val_mask, ['A1', 'A2']]
            .value_counts()
            .sort_index()
            .to_dict()
        )
        raise ValueError(
            "Validation ensemble failed to produce q22 predictions for some samples. "
            f"Missing cells: {uncovered_val_summary}"
        )

    q22_val_preds = q22_val_preds / q22_val_pred_counts


    # === 2.5 Weight Trimming (Option A: 1%/99% Trimming) ===
    def trim_weights(w, lower_p=1, upper_p=99):
        if len(w) > 0 and np.std(w) > 1e-6:
            low = np.percentile(w, lower_p)
            high = np.percentile(w, upper_p)
            return np.clip(w, low, high)
        return w

    print(f"Trimming q22 weights (5%/95% percentile)... Train mean: {np.mean(q22_train_oof):.4f}, Max: {np.max(q22_train_oof):.4f}")
    q22_train_oof = trim_weights(q22_train_oof, lower_p=5, upper_p=95)
    q22_val_preds = trim_weights(q22_val_preds, lower_p=5, upper_p=95)
    print(f"Post-trimming -> Train mean: {np.mean(q22_train_oof):.4f}, Max: {np.max(q22_train_oof):.4f}")

    # === 3. 第二/三步: 内外层交替优化 (Sequential Classification Learning) ===
    print("\n=== Step 2 & 3: Alternating Optimization for Policy Learning ===")
    
    Y2_array = df_train['Y2'].values
    A1_array = df_train['A1'].values
    A2_array = df_train['A2'].values
    
    grid_Q_full = np.unique(np.sort(Y2_array))
    # 限制搜索范围到合理的分位数区间，防止 q22 负值导致 grid search 跑到极端尾部
    q_lower = np.quantile(Y2_array, max(0.01, tau - 0.3))
    q_upper = np.quantile(Y2_array, min(0.99, tau + 0.3))
    grid_Q = grid_Q_full[(grid_Q_full >= q_lower) & (grid_Q_full <= q_upper)]

    epsilon_n = min(1e-4, 0.5 / np.sqrt(n_train))
    delta_n = min(1e-4, np.std(Y2_array) / (6 * np.sqrt(n_train)))
    hn = 0.2 / np.log(n_train)
    
    print(f"Alternating Optim Settings -> Grid size: {len(grid_Q)}, epsilon_n: {epsilon_n:.6f}, delta_n: {delta_n:.6f}, hn: {hn:.6f}")
    
    q_current = np.quantile(Y2_array, tau)
    f1, f2 = None, None
    best_sv = 0.0
    
    last_sign_f1 = None
    last_sign_f2 = None
    
    # 提前将验证不需要更新的张量放入目标设备，避免后续 SCL 迭代中几十次重复拷贝
    device_compute = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    H1_train_tensor = torch.tensor(df_train['Y0'].values, dtype=torch.float32).unsqueeze(1).to(device_compute)
    H2_train_tensor = torch.cat([
        torch.tensor(df_train['Y0'].values, dtype=torch.float32).unsqueeze(1),
        torch.tensor(df_train['Y1'].values, dtype=torch.float32).unsqueeze(1),
        torch.tensor(df_train['A1'].values, dtype=torch.float32).unsqueeze(1)
    ], dim=1).to(device_compute)
    
    for it in range(max_alt_iters):
        print(f"\n--- Alternating Optim Iteration {it+1}/{max_alt_iters} ---")
        print(f"Current q^(k-1) = {q_current:.6f}")
        
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
        
        # 提取策略符号计算 Zero-flip
        sign_f1 = (f1_train_out > 0).astype(int)
        sign_f2 = (f2_train_out > 0).astype(int)
        
        # Inner Level: Grid search over q
        q_new, sv_val = inner_optimization_grid(Y2_array, q22_train_oof, phi1, phi2, grid_Q, tau)
        
        best_sv = sv_val
        print(f"    -> Updated Empirical Survival Value (SV) at new q {q_new:.6f}: {sv_val:.6f} (Target: {1-tau:.6f})")
        
        # Convergence Checks
        policy_flip_count = 0
        if last_sign_f1 is not None and last_sign_f2 is not None:
            policy_flip_count = np.sum(sign_f1 != last_sign_f1) + np.sum(sign_f2 != last_sign_f2)
            
        print(f"    -> Convergence checks: |q_new - q_current| = {abs(q_new - q_current):.6f}, |SV - target| = {abs(sv_val - (1 - tau)):.6f}, Policy Flips: {policy_flip_count}")
        
        if abs(sv_val - (1 - tau)) <= epsilon_n:
            print(f"✅ AO Converged by epsilon: |{sv_val:.6f} - {1-tau:.6f}| <= {epsilon_n:.6f}")
            q_current = q_new
            break
            
        if abs(q_new - q_current) <= delta_n:
            print(f"✅ AO Converged by delta_n threshold (q settled): shifed {abs(q_new - q_current):.6f} <= {delta_n:.6f}")
            q_current = q_new
            break
            
        if it > 0 and policy_flip_count == 0:
            print(f"✅ AO Converged by Zero-Flip: Policies haven't changed from the last iteration.")
            q_current = q_new
            break
            
        q_current = q_new
        last_sign_f1 = sign_f1
        last_sign_f2 = sign_f2
        
    print(f"\n✅ Training Completed! Final optimal q: {q_current:.6f}, Estimated SV: {best_sv:.6f}")
    
    if save_models: # default: False
        save_trained_models(f1, f2, best_params, n_train, tau, phi_type, model_type, seed, df_train)
    
    return f1, f2, q_current, best_sv

def train_policy_prox_qtr_no_cf(n_train=1000, seed=20026, max_alt_iters=30, tau=0.5, phi_type=1, model_type="linear", save_models=False, dgp="S2"):
    """
    不带 Cross-Fitting (CF) 版本的策略学习函数。
    直接在全量训练集上估计一轮 q22 桥函数，然后进行策略学习。
    """
    if dgp == "S1":
        from Main.src.data_generate import data_gen, adjust_para_set_for_new_coding, origin_para_set
    elif dgp == "S2":
        from Main.src.data_generate_new import data_gen, adjust_para_set_for_new_coding, origin_para_set

    np.random.seed(seed)
    torch.manual_seed(seed)

    print("\n" + "="*50)
    print(f"🚀 Starting Proximal QTR Policy Learning (NO-CF MODE, {dgp})")
    
    n_val = int(n_train * 0.25)
    params = adjust_para_set_for_new_coding(origin_para_set)
    df_train = data_gen(n_train, params)
    df_val = data_gen(n_val, params)
    print(f"Generated data: Train({n_train}), Val({n_val})")
    
    # === 1. 滋扰估计: 一次性全量估计 q22 ===
    print("\n=== Step 1: Pre-estimating Bridge Functions (q22) w/o Cross-Fitting ===")
    
    q22_train_oof = np.zeros(len(df_train))
    q22_val_preds = np.zeros(len(df_val))
    
    # 为了调参和早停，简单拆分一次 80/20
    sub_train_full, sub_val_full = train_test_split(df_train, test_size=0.2, random_state=seed)
    
    for a1 in [1, -1]:
        for a2 in [1, -1]:
            if not ((sub_train_full['A1'] == a1) & (sub_train_full['A2'] == a2)).any():
                continue
            
            print(f"-> Estimating q22 for A1={a1}, A2={a2} using single split...")
            predict_q22_fn, _, _ = estimate_nuisance(sub_train_full, sub_val_full, a1, a2, n_trials=10)
            
            # 直接对全量 df_train 进行预测
            train_sub_mask = (df_train['A1'] == a1) & (df_train['A2'] == a2)
            if train_sub_mask.sum() > 0:
                q22_train_oof[train_sub_mask] = predict_q22_fn(df_train[train_sub_mask])
                
            # 对全量 df_val 进行预测
            val_sub_mask = (df_val['A1'] == a1) & (df_val['A2'] == a2)
            if val_sub_mask.sum() > 0:
                q22_val_preds[val_sub_mask] = predict_q22_fn(df_val[val_sub_mask])

    # === 2. Weight Trimming ===
    def trim_weights(w, lower_p=1, upper_p=99):
        if len(w) > 0 and np.std(w) > 1e-6:
            low = np.percentile(w, lower_p)
            high = np.percentile(w, upper_p)
            return np.clip(w, low, high)
        return w

    print(f"Trimming q22 weights (5%/95% percentile)... Train mean: {np.mean(q22_train_oof):.4f}, Max: {np.max(q22_train_oof):.4f}")
    q22_train_oof = trim_weights(q22_train_oof, lower_p=5, upper_p=95)
    q22_val_preds = trim_weights(q22_val_preds, lower_p=5, upper_p=95)

    # === 3. 内外层交替优化 (复用原版逻辑) ===
    print("\n=== Step 2 & 3: Alternating Optimization for Policy Learning ===")
    
    Y2_array = df_train['Y2'].values
    A1_array = df_train['A1'].values
    A2_array = df_train['A2'].values
    grid_Q_full = np.unique(np.sort(Y2_array))
    q_lower = np.quantile(Y2_array, max(0.01, tau - 0.3))
    q_upper = np.quantile(Y2_array, min(0.99, tau + 0.3))
    grid_Q = grid_Q_full[(grid_Q_full >= q_lower) & (grid_Q_full <= q_upper)]
    epsilon_n = min(1e-4, 0.5 / np.sqrt(n_train))
    delta_n = min(1e-4, np.std(Y2_array) / (6 * np.sqrt(n_train)))
    hn = 0.2 / np.log(n_train)
    
    q_current = np.quantile(Y2_array, tau)
    f1, f2 = None, None
    best_sv = 0.0
    last_sign_f1 = None
    last_sign_f2 = None
    
    device_compute = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    H1_train_tensor = torch.tensor(df_train['Y0'].values, dtype=torch.float32).unsqueeze(1).to(device_compute)
    H2_train_tensor = torch.cat([
        torch.tensor(df_train['Y0'].values, dtype=torch.float32).unsqueeze(1),
        torch.tensor(df_train['Y1'].values, dtype=torch.float32).unsqueeze(1),
        torch.tensor(df_train['A1'].values, dtype=torch.float32).unsqueeze(1)
    ], dim=1).to(device_compute)
    
    for it in range(max_alt_iters):
        print(f"\n--- Alternating Optim Iteration {it+1}/{max_alt_iters} ---")
        print(f"Current q^(k-1) = {q_current:.6f}")

        best_params = optimize_outer_hyperparams(df_train, q22_train_oof, df_val, q22_val_preds, 
                                                 q_current, n_trials=10, epochs=200, phi_type=phi_type, model_type=model_type)
        
        train_dataset = prepare_outer_tensors(df_train, q22_train_oof, q_current)
        val_dataset = prepare_outer_tensors(df_val, q22_val_preds, q_current)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        
        f1, f2, _ = train_outer_policies(train_loader, val_loader, best_params, phi_type=phi_type, model_type=model_type)
        
        f1.eval()
        f2.eval()
        with torch.no_grad():
            f1_out = f1(H1_train_tensor).cpu().numpy().flatten()
            f2_out = f2(H2_train_tensor).cpu().numpy().flatten()
            
        phi1 = stats.norm.cdf((A1_array * f1_out) / hn)
        phi2 = stats.norm.cdf((A2_array * f2_out) / hn)

        sign_f1 = (f1_out > 0).astype(int)
        sign_f2 = (f2_out > 0).astype(int)
        
        q_new, sv_val = inner_optimization_grid(Y2_array, q22_train_oof, phi1, phi2, grid_Q, tau)
        best_sv = sv_val
        print(f"    -> Updated Empirical Survival Value (SV) at new q {q_new:.6f}: {sv_val:.6f} (Target: {1-tau:.6f})")
        
        policy_flip_count = 0
        if last_sign_f1 is not None:
            policy_flip_count = np.sum(sign_f1 != last_sign_f1) + np.sum(sign_f2 != last_sign_f2)
            
        print(f"    -> Convergence checks: |q_new - q_current| = {abs(q_new - q_current):.6f}, |SV - target| = {abs(sv_val - (1 - tau)):.6f}, Policy Flips: {policy_flip_count}")
        
        if abs(sv_val - (1 - tau)) <= epsilon_n:
            print(f"✅ AO Converged by epsilon: |{sv_val:.6f} - {1-tau:.6f}| <= {epsilon_n:.6f}")
            q_current = q_new
            break
            
        if abs(q_new - q_current) <= delta_n:
            print(f"✅ AO Converged by delta_n threshold (q settled): shifed {abs(q_new - q_current):.6f} <= {delta_n:.6f}")
            q_current = q_new
            break
           
        if it > 0 and policy_flip_count == 0:
            print(f"✅ AO Converged by Zero-Flip: Policies haven't changed from the last iteration.")
            q_current = q_new
            break
            
        q_current = q_new
        last_sign_f1, last_sign_f2 = sign_f1, sign_f2

    print(f"\n✅ Training Completed! Final optimal q: {q_current:.6f}, Estimated SV: {best_sv:.6f}")
        
    if save_models:
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
