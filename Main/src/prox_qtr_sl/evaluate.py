import os
import torch
import numpy as np
import pandas as pd
import argparse

from Main.src.prox_qtr_sl.step2_inner import inner_optimization
from Main.src.prox_qtr_sl.step3_outer import Policy_Linear, Policy_NN, psi
from Main.src.deepmmr.MMR_model import MLP_for_MMR
from Main.src.data_generate import data_gen, adjust_para_set_for_new_coding, origin_para_set

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_policy_model(model_path, model_type, input_dim):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    params = checkpoint['hyperparams']
    
    if model_type == "linear":
        model = Policy_Linear(input_dim)
    elif model_type == "nn":
        model = Policy_NN(input_dim, hidden_dim=params['hidden_dim'], num_layers=params['num_layers'])
    else:
        raise ValueError("Invalid model_type")
        
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model

def load_q22_model(model_path):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    params = checkpoint['hyperparams']
    
    input_dim = 4 # Z1, Z2, Y0, Y1
    model = MLP_for_MMR(input_dim, params['network_width'], params['network_depth'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model

def offline_evaluation(df_test: pd.DataFrame, config_str: str, tau: float = 0.5, phi_type: int = 1, model_type: str = "linear"):
    # 模型统一保存在 Main/models 目录下 (当前文件在 Main/src/prox_qtr_sl/)
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
    
    # 1. Load Policies
    f1_path = os.path.join(models_dir, f"f1_{config_str}.pt")
    f2_path = os.path.join(models_dir, f"f2_{config_str}.pt")
    if not os.path.exists(f1_path) or not os.path.exists(f2_path):
        print(f"Error: Could not find policy models at {f1_path} or {f2_path}")
        return None, None
        
    f1 = load_policy_model(f1_path, model_type, input_dim=1)
    f2 = load_policy_model(f2_path, model_type, input_dim=3)
    
    # 2. Predict Action Policies
    print("-> Infusing test causal history to policy machines...")
    with torch.no_grad():
        H1_test = torch.tensor(df_test['Y0'].values, dtype=torch.float32).unsqueeze(1).to(device)
        H2_test = torch.cat([torch.tensor(df_test['Y0'].values, dtype=torch.float32).unsqueeze(1),
                              torch.tensor(df_test['Y1'].values, dtype=torch.float32).unsqueeze(1),
                              torch.tensor(df_test['A1'].values, dtype=torch.float32).unsqueeze(1)], dim=1).to(device)
                              
        f1_out = f1(H1_test).cpu().numpy().flatten()
        f2_out = f2(H2_test).cpu().numpy().flatten()
        
        d1_pred = np.sign(f1_out)
        d1_pred[d1_pred == 0] = 1
        d2_pred = np.sign(f2_out)
        d2_pred[d2_pred == 0] = 1
        
    # 3. Predict q22 nuisance for test set instances respecting their observed actions
    print("-> Reconstructing q22 predictions for out-of-sample combinations...")
    q22_test_preds = np.zeros(len(df_test))
    
    # 解析出 n_train 的数量，因为 q22_path 指定使用了 n_train 进行独立命名
    n_train = config_str.split("_")[0].replace("ntrain", "")
    for a1 in [1, -1]:
        for a2 in [1, -1]:
            q22_path = os.path.join(models_dir, f"q22_a1_{a1}_a2_{a2}_{n_train}.pt")
            
            mask = (df_test['A1'] == a1) & (df_test['A2'] == a2)
            if mask.sum() == 0:
                continue
                
            if not os.path.exists(q22_path):
                print(f"Warning: q22 model {q22_path} is missing! Defaulting partial weights to 0.")
                continue
                
            model_q22 = load_q22_model(q22_path)
            sub_df = df_test[mask]
            with torch.no_grad():
                Z1 = torch.tensor(sub_df['Z1'].values, dtype=torch.float32).unsqueeze(1).to(device)
                Z2 = torch.tensor(sub_df['Z2'].values, dtype=torch.float32).unsqueeze(1).to(device)
                Y0 = torch.tensor(sub_df['Y0'].values, dtype=torch.float32).unsqueeze(1).to(device)
                Y1 = torch.tensor(sub_df['Y1'].values, dtype=torch.float32).unsqueeze(1).to(device)
                
                preds = model_q22(torch.cat([Z1, Z2, Y0, Y1], dim=1)).cpu().numpy().flatten()
                
            q22_test_preds[sub_df.index] = preds

    # 4. Calculate Quantile-Value Function (QV)
    print("-> Deriving Empirical Quantile-Value Function...")
    optimal_q = inner_optimization(df_test['Y2'], df_test['A1'], df_test['A2'], 
                                   d1_pred, d2_pred, q22_test_preds, tau=tau)
                                   
    # 5. Calculate Survival Value Function (SV_psi)
    with torch.no_grad():
        b_I = torch.tensor((df_test['Y2'].values > optimal_q), dtype=torch.float32).to(device)
        b_q22 = torch.tensor(q22_test_preds, dtype=torch.float32).to(device)
        b_A1 = torch.tensor(df_test['A1'].values, dtype=torch.float32).to(device)
        b_A2 = torch.tensor(df_test['A2'].values, dtype=torch.float32).to(device)
        t_f1_out = torch.tensor(f1_out, dtype=torch.float32).to(device)
        t_f2_out = torch.tensor(f2_out, dtype=torch.float32).to(device)
        
        psi_eval = psi(b_A1 * t_f1_out, b_A2 * t_f2_out, phi_type)
        sv_psi = torch.mean(b_I * b_q22 * psi_eval).item()
        
    print(f"\n==========================================")
    print(f"         OFFLINE EVALUATION RESULTS       ")
    print(f"==========================================")
    print(f"CONFIG: {config_str}")
    print(f"ESTIMATED QUANTILE-VALUE (QV_tau={tau}): {optimal_q:.6f}")
    print(f"ESTIMATED SURVIVAL-VALUE (SV_psi):    {sv_psi:.6f}")
    print(f"==========================================\n")
    return optimal_q, sv_psi

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Evaluation for Proximal QTR")
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--phi_type", type=int, default=1)
    parser.add_argument("--model_type", type=str, default="linear")
    parser.add_argument("--seed", type=int, default=20026)
    parser.add_argument("--n_test", type=int, default=1000)
    args = parser.parse_args()
    
    config_str = f"ntrain{args.n_train}_tau{args.tau}_phi{args.phi_type}_model{args.model_type}_seed{args.seed}"
    params = adjust_para_set_for_new_coding(origin_para_set)
    df_test = data_gen(args.n_test, params)
    
    offline_evaluation(df_test, config_str, args.tau, args.phi_type, args.model_type)
