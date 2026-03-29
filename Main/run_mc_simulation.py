import os
import json
import torch
import numpy as np
import argparse
import pandas as pd

from Main.src.data_generate import dynamic_intervened_data_gen, intervened_data_gen, adjust_para_set_for_new_coding, origin_para_set
from Main.src.qtr_biopt_sl.estimate import train_policy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_monte_carlo(n_train=1000, seed=2026, K_folds=2, max_alt_iters=3, tau=0.5, phi_type=1, model_type="linear", 
                     mc_reps=100):
    """
    Monte Carlo 模拟：重复 B 次 (mc_reps) 实验。
    每次实验：
    1. 生成新的训练/验证集。
    2. 训练出策略 f1, f2。
    3. 生成极大量的独立反事实数据，计算该策略的“真实”分位数值。
    最后统计均值和标准差。
    """
    params = adjust_para_set_for_new_coding(origin_para_set)
    # 强制设定 mc_n 为训练集大小的 10 倍
    mc_sample_size = n_train * 10
    
    q_values = []
    print(f"🚀 Starting Monte Carlo Simulation: Total {mc_reps} reps...")
    print(f"   Settings: n_train={n_train}, tau={tau}, phi={phi_type}, model={model_type}")
    
    for i in range(mc_reps):
        current_seed = seed + i
        print(f"\n>>> MC Repetition {i+1}/{mc_reps} (Seed: {current_seed})")
        
        # 1. 重新训练模型 (不保存模型文件以节省磁盘 IO)
        f1, f2, q_train_est, _ = train_policy(
            n_train=n_train, seed=current_seed, K_folds=K_folds, 
            max_alt_iters=max_alt_iters, tau=tau, 
            phi_type=phi_type, model_type=model_type, save_models=False
        )
        
        # 2. 生成极大量的独立评估数据 (使用独立 seed 确保评估的公平性)
        # 用训练好的模型指导干预
        df_mc = dynamic_intervened_data_gen(mc_sample_size, params, f1=f1, f2=f2, device=device)
        
        # 3. 计算在介入分布下的真实分位数值
        mc_q = np.quantile(df_mc['Y2'], tau)
        q_values.append(mc_q)
        print(f"    Result for Rep {i+1}: True Q_{tau} = {mc_q:.4f}")
            
    q_mean = np.mean(q_values)
    q_std = np.std(q_values)
    
    print("\n" + "="*50)
    print(f"      FINAL MONTE CARLO SUMMARY ({mc_reps} REPS)     ")
    print("="*50)
    
    # 无论是否进行指标对比，首先汇报DGP过程算出的真值 (True)
    print(f"Estimating Policy Ceiling (Always Treat)...")
    df_true = intervened_data_gen(mc_sample_size * 10, params, a=[1, 1])
    true_q = np.quantile(df_true['Y2'], tau)
    print(f"True Quantile:      {true_q:.4f}")
    
    print("-" * 30)
    print(f"Model Mean True Quantile:  {q_mean:.4f}")
    print(f"Model Standard Deviation:   {q_std:.4f}")
    
    # 计算 Regret 和 RMSE (现在强制汇报)
    regret = true_q - q_mean
    rmse = np.sqrt(np.mean((true_q - np.array(q_values))**2))
    
    print("-" * 30)
    print(f"True Regret (True- Mean by estimated policies): {regret:.4f}")
    print(f"RMSE (Root Mean Sq Error): {rmse:.4f}")
        
    print("="*50 + "\n")
    return true_q, q_mean, q_std, regret, rmse


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
