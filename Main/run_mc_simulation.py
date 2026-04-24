import os
import json
import torch
import numpy as np
import argparse
import pandas as pd

from Main.src.data_generate import dynamic_intervened_data_gen, intervened_data_gen, origin_para_set
from Main.src.prox_qtr_sl.estimate_prox_qtr_sl import train_policy_prox_qtr_sl
from Main.src.SRA.estimate_SRA import train_policy_SRA
from Main.src.Oracle.estimate_Oracle import train_policy_Oracle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_monte_carlo(n_train=1000, seed=20026, K_folds=2, max_alt_iters=10, tau=0.5, phi_type=1, model_type="linear",
                     mc_reps=100, scenario="S1"):
    """
    Monte Carlo 模拟：重复 B 次 (mc_reps) 实验，比较 Proximal QTR, SRA 和 Oracle。
    """
    params = origin_para_set
    mc_sample_size = 100000
    
    # 提前计算 DGP 的真实上限 (Always Treat [1,1]) 以作为基准
    print(f"\nEstimating Policy Ceiling (Always Treat)...")
    df_true = intervened_data_gen(mc_sample_size * 20, params, a=[1, 1], scenario=scenario)
    true_q = np.quantile(df_true['Y2'], tau)
    
    # 初始化三个方法的存储列表: q_true_(of_est) 用于计算后悔; q_train_est 用于计算 Overall Error
    q_values_prox = []
    q_train_ests_prox = []
    
    q_values_sra = []
    q_train_ests_sra = []
    
    q_values_oracle = []
    q_train_ests_oracle = []
    
    print(f"🚀 Starting Comparative Monte Carlo Simulation: Total {mc_reps} reps...")
    print(f"   Settings: n_train={n_train}, tau={tau}, phi={phi_type}, model={model_type}")
    print(f"   True Upper Bound (Optimal Quantile): {true_q:.6f}")
    
    for i in range(mc_reps):
        current_seed = seed + i
        print(f"\n>>> MC Repetition {i+1}/{mc_reps} (Seed: {current_seed})")
        
        # --- 1. Proximal QTR ---
        print("    [Method] Proximal QTR...")
        f1_p, f2_p, q_est_p, _ = train_policy_prox_qtr_sl(
            n_train=n_train, seed=current_seed, K_folds=K_folds, 
            max_alt_iters=max_alt_iters, tau=tau, 
            phi_type=phi_type, model_type=model_type, save_models=False, dgp=scenario
        )
        df_mc_p = dynamic_intervened_data_gen(mc_sample_size, params, f1=f1_p, f2=f2_p, device=device, scenario=scenario)
        q_values_prox.append(np.quantile(df_mc_p['Y2'], tau))
        q_train_ests_prox.append(q_est_p)
        
        # --- 2. SRA Estimator ---
        print("    [Method] SRA (Logistic)...")
        f1_s, f2_s, q_est_s, _ = train_policy_SRA(
            n_train=n_train, seed=current_seed, tau=tau, 
            phi_type=phi_type, model_type=model_type, save_models=False, dgp=scenario
        )
        df_mc_s = dynamic_intervened_data_gen(mc_sample_size, params, f1=f1_s, f2=f2_s, device=device, scenario=scenario)
        q_values_sra.append(np.quantile(df_mc_s['Y2'], tau))
        q_train_ests_sra.append(q_est_s)
        
        # --- 3. Oracle Estimator ---
        print("    [Method] Oracle (Unbiased)...")
        f1_o, f2_o, q_est_o, _ = train_policy_Oracle(
            n_train=n_train, seed=current_seed, tau=tau, 
            phi_type=phi_type, model_type=model_type, save_models=False, dgp=scenario
        )
        df_mc_o = dynamic_intervened_data_gen(mc_sample_size, params, f1=f1_o, f2=f2_o, device=device, scenario=scenario)
        q_values_oracle.append(np.quantile(df_mc_o['Y2'], tau))
        q_train_ests_oracle.append(q_est_o)
        
        print(f"    Rep {i+1} True Quantiles (Performance): Prox={q_values_prox[-1]:.6f}, SRA={q_values_sra[-1]:.6f}, Oracle={q_values_oracle[-1]:.6f}")
        print(f"    Rep {i+1} Estimated Quantiles (Training): Prox={q_train_ests_prox[-1]:.6f}, SRA={q_train_ests_sra[-1]:.6f}, Oracle={q_train_ests_oracle[-1]:.6f}")
            
    # 结果汇总计算
    methods = ["Proximal", "SRA", "Oracle"]
    q_data_perf = [np.array(q_values_prox), np.array(q_values_sra), np.array(q_values_oracle)]
    q_data_ests = [np.array(q_train_ests_prox), np.array(q_train_ests_sra), np.array(q_train_ests_oracle)]
    
    summary_list = []
    for name, vals_perf, vals_ests in zip(methods, q_data_perf, q_data_ests):
        m_q = np.mean(vals_perf)
        std_q = np.std(vals_perf)
        regret = true_q - m_q
        rmse = np.sqrt(np.mean((true_q - vals_perf)**2))
        # 新增指标: Overall Error |V - V_hat|
        overall_error = np.mean(np.abs(true_q - vals_ests))
        
        summary_list.append({
            "Method": name, "Mean_Q": m_q, "Std_Q": std_q, "Regret": regret, "RMSE": rmse, "Overall_Err": overall_error
        })

    # 保存结果到 CSV
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'mc_compara')
    os.makedirs(results_dir, exist_ok=True)
    
    file_name = f"compara_ntrain{n_train}_tau{tau}_phi{phi_type}_model{model_type}.csv"
    save_path = os.path.join(results_dir, file_name)
    
    df_res = pd.DataFrame(summary_list)
    df_res.to_csv(save_path, index=False)
    print(f"📊 Comparison results successfully saved to: {save_path}")

    # 打印对比表格
    print("\n" + "="*110)
    print(f"      COMPARATIVE MONTE CARLO SUMMARY ({mc_reps} REPS) - True Upper bound: {true_q:.6f}     ")
    print("="*110)
    # Header format updated to include "Overall Err"
    print(f"{'Method':<15} | {'Mean Q':<13} | {'Std Q':<13} | {'Regret':<13} | {'RMSE':<13} | {'Overall Err':<13}")
    print("-" * 110)
    for s in summary_list:
        print(f"{s['Method']:<15} | {s['Mean_Q']:<13.6f} | {s['Std_Q']:<13.6f} | {s['Regret']:<13.6f} | {s['RMSE']:<13.6f} | {s['Overall_Err']:<13.6f}")
    print("="*110 + "\n")
    
    # 返回 Proximal 的结果，新增一项
    prox_res = summary_list[0]
    return true_q, prox_res['Mean_Q'], prox_res['Std_Q'], prox_res['Regret'], prox_res['RMSE'], prox_res['Overall_Err']


def run_parameter_grid_analysis(n_reps:int):
    # 模拟设置
    n_train_list = [500, 1000, 2000, 5000]
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
                
                # 运行 MC 仿真 (增加 Overall Error)
                true_q, q_mean, q_std, regret, rmse, overall_err = run_monte_carlo(
                    n_train=n_train, seed=seed, tau=tau, 
                    phi_type=p_type, model_type=m_type, 
                    mc_reps=mc_reps, scenario="S1"
                )
                
                # 记录结果 (直接使用返回值)
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
                    "rmse": float(rmse),
                    "overall_error": float(overall_err)
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
    parser.add_argument("--n_train", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=285063)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--phi_type", type=int, default=1)
    parser.add_argument("--model_type", type=str, default="linear")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--reps", type=int, default=10, help="Number of MC repetitions")
    parser.add_argument("--scenario", type=str, choices=["S1", "S2"], default="S1")
    parser.add_argument("--grid", action="store_true", help="Run full parameter grid analysis instead of single run")
    args = parser.parse_args()
    
    if args.grid:
        run_parameter_grid_analysis(n_reps=50)
    else:
        run_monte_carlo(
            n_train=args.n_train, seed=args.seed, K_folds=args.folds, 
            tau=args.tau, phi_type=args.phi_type, model_type=args.model_type,
            mc_reps=args.reps, scenario=args.scenario
        )
