import os
import torch
import numpy as np
import argparse

from Main.src.data_generate import dynamic_intervened_data_gen, intervened_data_gen, adjust_para_set_for_new_coding, origin_para_set
from Main.estimate import train_policy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_monte_carlo(n_train=1000, seed=2026, K_folds=2, max_alt_iters=3, tau=0.5, phi_type=1, model_type="linear", 
                     mc_reps=100, compare_oracle=True):
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
    
    # 无论是否进行指标对比，首先汇报上帝视角的真值 (Oracle)
    print(f"[Oracle Context] Estimating Policy Ceiling (Always Treat)...")
    df_oracle = intervened_data_gen(mc_sample_size * 5, params, a=[1, 1])
    oracle_q = np.quantile(df_oracle['Y2'], tau)
    print(f"Oracle True Quantile:      {oracle_q:.4f}")
    
    print("-" * 30)
    print(f"Model Mean True Quantile:  {q_mean:.4f}")
    print(f"Model Standard Deviation:   {q_std:.4f}")
    
    if compare_oracle:
        # 计算 Regret 和 RMSE
        regret = oracle_q - q_mean
        rmse = np.sqrt(np.mean((oracle_q - np.array(q_values))**2))
        
        print("-" * 30)
        print(f"True Regret (Oracle-Mean): {regret:.4f}")
        print(f"RMSE (Root Mean Sq Error): {rmse:.4f}")
        
    print("="*50 + "\n")
    return q_mean, q_std, q_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo Estimator Performance Simulator")
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20026)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--phi_type", type=int, default=1)
    parser.add_argument("--model_type", type=str, default="linear")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--reps", type=int, default=10, help="Number of MC repetitions")
    parser.add_argument("--no_compare", action="store_true", help="Skip Oracle comparison")
    args = parser.parse_args()
    
    run_monte_carlo(
        n_train=args.n_train, seed=args.seed, K_folds=args.folds, 
        tau=args.tau, phi_type=args.phi_type, model_type=args.model_type,
        mc_reps=args.reps, compare_oracle=not args.no_compare
    )
