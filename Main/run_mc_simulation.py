import os
import json
import torch
import numpy as np
import argparse
import pandas as pd

from Main.src.data_generate import dynamic_intervened_data_gen, intervened_data_gen, adjust_para_set_for_new_coding, origin_para_set
from Main.src.prox_qtr_sl.estimate_prox_qtr_sl import train_policy_prox_qtr_sl
from Main.src.SRA.estimate_SRA import train_policy_SRA
from Main.src.Oracle.estimate_Oracle import train_policy_Oracle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_monte_carlo(n_train=1000, seed=2026, K_folds=2, max_alt_iters=3, tau=0.5, phi_type=1, model_type="linear", 
                     mc_reps=100):
    """
    Monte Carlo 模拟：重复 B 次 (mc_reps) 实验，比较 Proximal QTR, SRA 和 Oracle。
    """
    params = adjust_para_set_for_new_coding(origin_para_set)
    mc_sample_size = n_train * 10
    
    # 初始化三个方法的存储列表
    q_values_prox = []
    q_values_sra = []
    q_values_oracle = []
    
    print(f"🚀 Starting Comparative Monte Carlo Simulation: Total {mc_reps} reps...")
    print(f"   Settings: n_train={n_train}, tau={tau}, phi={phi_type}, model={model_type}")
    
    for i in range(mc_reps):
        current_seed = seed + i
        print(f"\n>>> MC Repetition {i+1}/{mc_reps} (Seed: {current_seed})")
        
        # --- 1. Proximal QTR ---
        print("    [Method] Proximal QTR...")
        f1_p, f2_p, _, _ = train_policy_prox_qtr_sl(
            n_train=n_train, seed=current_seed, K_folds=K_folds, 
            max_alt_iters=max_alt_iters, tau=tau, 
            phi_type=phi_type, model_type=model_type, save_models=False
        )
        df_mc_p = dynamic_intervened_data_gen(mc_sample_size, params, f1=f1_p, f2=f2_p, device=device)
        q_values_prox.append(np.quantile(df_mc_p['Y2'], tau))
        
        # --- 2. SRA Estimator ---
        print("    [Method] SRA (Logistic)...")
        f1_s, f2_s, _, _ = train_policy_SRA(
            n_train=n_train, seed=current_seed, tau=tau, 
            phi_type=phi_type, model_type=model_type, save_models=False
        )
        df_mc_s = dynamic_intervened_data_gen(mc_sample_size, params, f1=f1_s, f2=f2_s, device=device)
        q_values_sra.append(np.quantile(df_mc_s['Y2'], tau))
        
        # --- 3. Oracle Estimator ---
        print("    [Method] Oracle (Unbiased)...")
        f1_o, f2_o, _, _ = train_policy_Oracle(
            n_train=n_train, seed=current_seed, tau=tau, 
            phi_type=phi_type, model_type=model_type, save_models=False
        )
        df_mc_o = dynamic_intervened_data_gen(mc_sample_size, params, f1=f1_o, f2=f2_o, device=device)
        q_values_oracle.append(np.quantile(df_mc_o['Y2'], tau))
        
        print(f"    Rep {i+1} Results: Prox={q_values_prox[-1]:.4f}, SRA={q_values_sra[-1]:.4f}, Oracle={q_values_oracle[-1]:.4f}")
            
    # 计算 DGP 的真实上限 (Always Treat [1,1])
    print(f"\nEstimating Policy Ceiling (Always Treat)...")
    df_true = intervened_data_gen(mc_sample_size * 20, params, a=[1, 1])
    true_q = np.quantile(df_true['Y2'], tau)

    # 结果汇总计算
    methods = ["Proximal", "SRA", "Oracle"]
    q_data = [np.array(q_values_prox), np.array(q_values_sra), np.array(q_values_oracle)]
    
    summary_list = []
    for name, vals in zip(methods, q_data):
        m_q = np.mean(vals)
        std_q = np.std(vals)
        regret = true_q - m_q
        rmse = np.sqrt(np.mean((true_q - vals)**2))
        summary_list.append({
            "Method": name, "Mean_Q": m_q, "Std_Q": std_q, "Regret": regret, "RMSE": rmse
        })

    # 打印对比表格
    print("\n" + "="*85)
    print(f"      COMPARATIVE MONTE CARLO SUMMARY ({mc_reps} REPS) - True Upper bound: {true_q:.4f}     ")
    print("="*85)
    print(f"{'Method':<15} | {'Mean Q':<12} | {'Std Q':<12} | {'Regret':<12} | {'RMSE':<12}")
    print("-" * 85)
    for s in summary_list:
        print(f"{s['Method']:<15} | {s['Mean_Q']:<12.4f} | {s['Std_Q']:<12.4f} | {s['Regret']:<12.4f} | {s['RMSE']:<12.4f}")
    print("="*85 + "\n")
    
    # 为了保持外部调用的兼容性，默认返回 Proximal 的结果
    prox_res = summary_list[0]
    return true_q, prox_res['Mean_Q'], prox_res['Std_Q'], prox_res['Regret'], prox_res['RMSE']


def run_parameter_grid_analysis(n_reps:int):
    # 模拟设置
    n_train_list = [5000, 10000] # 500, 2000
    phi_types = [1, 2, 3, 4]
    model_types = ["linear", "nn"]
    tau = 0.5
    seed = 20026
    mc_reps = n_reps
    
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'mc_res')
    os.makedirs(results_dir, exist_ok=True)
    
    summary_data = []

    print(f"🌟 Starting Comprehensive Parameter Analysis across {len(n_train_list)} scale steps...")
    
    for n_train in n_train_list:
        print(f"\n##########################################")
        print(f"  SCALE STEP: N_TRAIN = {n_train}")
        print(f"##########################################")
        
        for m_type in model_types:
            for p_type in phi_types:
                config_name = f"ntrain{n_train}_tau{tau}_phi{p_type}_model{m_type}"
                result_file = os.path.join(results_dir, f"{config_name}.json")
                
                print(f"\n[Running] {config_name}...")
                
                # 运行 MC 仿真 (直接获取true、均值、标准差、Regret 和 RMSE)
                true_q, q_mean, q_std, regret, rmse = run_monte_carlo(
                    n_train=n_train, seed=seed, tau=tau, 
                    phi_type=p_type, model_type=m_type, 
                    mc_reps=mc_reps
                )
                
                # 记录结果 (直接使用返回值，不再重复采样计算)
                res_dict = {
                    "n_train": n_train,
                    "tau": tau,
                    "phi_type": p_type,
                    "model_type": m_type,
                    "mc_reps": mc_reps,
                    "true_q": float(true_q),
                    "mean_q": float(q_mean),
                    "std_q": float(q_std),
                    "regret": float(regret),
                    "rmse": float(rmse)
                }
                
                with open(result_file, 'w') as f:
                    json.dump(res_dict, f, indent=4)
                
                summary_data.append(res_dict)

    # 导出汇总 CSV 方便绘图
    df_results = pd.DataFrame(summary_data)
    df_results.to_csv(os.path.join(results_dir, "full_summary.csv"), index=False)
    print(f"\n✅ All grid experiments finished. Summary saved to results/mc_res/full_summary.csv")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo Estimator Performance Simulator")
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20026)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--phi_type", type=int, default=1)
    parser.add_argument("--model_type", type=str, default="linear")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--reps", type=int, default=10, help="Number of MC repetitions")
    parser.add_argument("--grid", action="store_true", help="Run full parameter grid analysis instead of single run")
    args = parser.parse_args()
    
    if args.grid:
        run_parameter_grid_analysis(n_reps=50)
    else:
        run_monte_carlo(
            n_train=args.n_train, seed=args.seed, K_folds=args.folds, 
            tau=args.tau, phi_type=args.phi_type, model_type=args.model_type,
            mc_reps=args.reps
        )
