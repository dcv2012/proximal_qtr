import numpy as np
import pandas as pd
import argparse
import os
import time
import torch
import warnings
warnings.filterwarnings("ignore")

from Main.src.prox_qtr_sl.estimate_prox_qtr_sl import train_policy_prox_qtr_sl
from Main.src.SRA.estimate_SRA import train_policy_SRA
from Main.src.Oracle.estimate_Oracle import train_policy_Oracle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_arguments():
    parser = argparse.ArgumentParser(description="Monte Carlo Comparative Analysis for Proximal QTR, SRA, and Oracle")
    
    # 手动可调数据集参数/调参设定
    parser.add_argument("--n_train", type=int, default=1000, help="Training set sample size (e.g. 500, 1000, 2000, 5000)")
    parser.add_argument("--mc_reps", type=int, default=100, help="Number of Monte Carlo repetitions")
    parser.add_argument("--mc_eval_size", type=int, default=100000, help="Size of data generated for policy evaluation")
    parser.add_argument("--seed", type=int, default=20026, help="Base random seed for data generation and model initializations")
    
    # 策略模型参数
    parser.add_argument("--tau", type=float, default=0.5, help="Target quantile level (e.g., 0.5 for median)")
    parser.add_argument("--phi_type", type=int, choices=[0, 1, 2, 3, 4], default=1, help="Type of surrogate loss function")
    parser.add_argument("--model_type", type=str, choices=["linear", "nn"], default="linear", help="Class of policy function U_n")
    parser.add_argument("--k_folds", type=int, default=2, help="Number of folds for cross-fitting")
    parser.add_argument("--max_alt_iters", type=int, default=20, help="Max iterations for SCL optimization")
    parser.add_argument("--dgp", type=str, choices=["S1", "S2"], default="S2", help="Data generation process version to use (S1=data_generate, S2=data_generate_new)")
    
    return parser.parse_args()


def plot_boxplot(results, args, res_dir):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Prepare data for plotting
        plot_data = []
        for method in ["Proximal", "SRA", "Oracle"]:
            for val in results[method]["true_perf"]:
                plot_data.append({"Estimator": method, "V(d_hat)": val})
                
        df_plot = pd.DataFrame(plot_data)
        
        plt.figure(figsize=(8, 6))
        sns.boxplot(x="Estimator", y="V(d_hat)", data=df_plot, palette="Set2")
        plt.title(f"True Quantile Value of Estimated Policies\n(n={args.n_train}, tau={args.tau}, reps={args.mc_reps})")
        plt.ylabel(r"Target Quantile Value $V(\hat{d})$")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        fname = f"boxplot_n{args.n_train}_tau{args.tau}_phi{args.phi_type}_{args.model_type}_reps{args.mc_reps}.png"
        save_path = os.path.join(res_dir, fname)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Boxplot successfully saved to: {save_path}")
        
    except ImportError:
        print("\n⚠️ Matplotlib or Seaborn is not installed. Skipping boxplot generation.")
        print("To generate plots, please run: pip install matplotlib seaborn")


def run_comparative_mc(args):
    print("\n" + "#"*70)
    print("   MONTE CARLO COMPARATIVE ANALYSIS (Proximal vs SRA vs Oracle)")
    print("#"*70)
    print(f"Configurations:")
    for k, v in vars(args).items():
        print(f" - {k}: {v}")
    print(f" - Device: {device}\n")
    
    
    if args.dgp == "S1":
        from Main.src.data_generate import dynamic_intervened_data_gen, adjust_para_set_for_new_coding, origin_para_set
    else:
        from Main.src.data_generate_new import dynamic_intervened_data_gen, adjust_para_set_for_new_coding, origin_para_set
    
    params = adjust_para_set_for_new_coding(origin_para_set)
    mc_sample_size = args.mc_eval_size
    
    # 记录追踪数组
    # true_perf: 模型产生的最优策略的实际性能 -> V(d_hat)
    # train_est: 训练时预估出的最优分位值 -> \hat{V}(d_hat)
    results = {
        "Proximal": {"true_perf": [], "train_est": []},
        "SRA": {"true_perf": [], "train_est": []},
        "Oracle": {"true_perf": [], "train_est": []}
    }
    
    # 提前定义好保存路径并在开始前写入由于随时追踪原始结果的文件头
    res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'comparative_analysis')
    os.makedirs(res_dir, exist_ok=True)
    fname = f"comparative_n{args.n_train}_tau{args.tau}_phi{args.phi_type}_{args.model_type}_{args.dgp}_reps{args.mc_reps}.csv"
    save_path = os.path.join(res_dir, fname)
    
    with open(save_path, 'w') as f:
        f.write("Repetition,Estimator,True_Perf,Train_Est\n")
    
    st_time = time.time()
    
    for i in range(args.mc_reps):
        current_seed = args.seed + i
        print(f"\n[{i+1}/{args.mc_reps}] MC Repetition Seed: {current_seed}")
        
        # === 1. Proximal QTR ===
        print("  -> Training Proximal QTR...")
        f1_p, f2_p, q_est_p, _ = train_policy_prox_qtr_sl(
            n_train=args.n_train, seed=current_seed, K_folds=args.k_folds, 
            max_alt_iters=args.max_alt_iters, tau=args.tau, 
            phi_type=args.phi_type, model_type=args.model_type, save_models=False,
            dgp=args.dgp
        )
        df_eval_p = dynamic_intervened_data_gen(mc_sample_size, params, f1=f1_p, f2=f2_p, device=device)
        results["Proximal"]["true_perf"].append(np.quantile(df_eval_p['Y2'], args.tau))
        results["Proximal"]["train_est"].append(q_est_p)
        
        # === 2. SRA Estimator ===
        print("  -> Training SRA Estimator...")
        f1_s, f2_s, q_est_s, _ = train_policy_SRA(
            n_train=args.n_train, seed=current_seed, K_folds=args.k_folds,
            max_alt_iters=args.max_alt_iters, tau=args.tau, 
            phi_type=args.phi_type, model_type=args.model_type, save_models=False,
            dgp=args.dgp
        )
        df_eval_s = dynamic_intervened_data_gen(mc_sample_size, params, f1=f1_s, f2=f2_s, device=device)
        results["SRA"]["true_perf"].append(np.quantile(df_eval_s['Y2'], args.tau))
        results["SRA"]["train_est"].append(q_est_s)
        
        # === 3. Oracle Estimator ===
        print("  -> Training Oracle Estimator...")
        f1_o, f2_o, q_est_o, _ = train_policy_Oracle(
            n_train=args.n_train, seed=current_seed, K_folds=args.k_folds,
            max_alt_iters=args.max_alt_iters, tau=args.tau, 
            phi_type=args.phi_type, model_type=args.model_type, save_models=False,
            dgp=args.dgp
        )
        df_eval_o = dynamic_intervened_data_gen(mc_sample_size, params, f1=f1_o, f2=f2_o, device=device)
        results["Oracle"]["true_perf"].append(np.quantile(df_eval_o['Y2'], args.tau))
        results["Oracle"]["train_est"].append(q_est_o)
        
        # Debug Report Output for the repetition
        iter_log = (
            f"   Perf  -> Prox: {results['Proximal']['true_perf'][-1]:.6f} | "
            f"SRA: {results['SRA']['true_perf'][-1]:.6f} | "
            f"Oracle: {results['Oracle']['true_perf'][-1]:.6f}"
        )
        print(iter_log)
        
        # 每次循环结束后随时将结果追加记录到文件
        with open(save_path, 'a') as f:
            f.write(f"{i+1},Proximal,{results['Proximal']['true_perf'][-1]:.6f},{results['Proximal']['train_est'][-1]:.6f}\n")
            f.write(f"{i+1},SRA,{results['SRA']['true_perf'][-1]:.6f},{results['SRA']['train_est'][-1]:.6f}\n")
            f.write(f"{i+1},Oracle,{results['Oracle']['true_perf'][-1]:.6f},{results['Oracle']['train_est'][-1]:.6f}\n")
        
    print(f"\n[Simulation Completed] Execution Time: {(time.time() - st_time)/60:.2f} minutes")
    
    # 统计指标 (按 4.3 Section 的三条标准计算)
    summary_data = []
    for method in ["Proximal", "SRA", "Oracle"]:
        perf_array = np.array(results[method]["true_perf"])
        est_array = np.array(results[method]["train_est"])
        
        # 1. Mean of true quantile value estimated by the model 
        mean_v_true = np.mean(perf_array)
        
        # 2. Standard deviation of above mean
        std_v_true = np.std(perf_array)
        
        # 3. Mean absolute estimation error (Generalization bound): |V(d_hat) - V_hat(d_hat)|
        # 这是为了反映每种模型对自身产生的策略价值评估的准确性（去除了理论最优上界的依赖）
        estimation_error = np.mean(np.abs(perf_array - est_array))
        
        summary_data.append({
            "Estimator": method,
            "Mean_True_Q_V(d)": mean_v_true,
            "Std_True_Q_V(d)": std_v_true,
            "Estimation_Error_Mean": estimation_error
        })
        
    df_summary = pd.DataFrame(summary_data)
    
    # Print the specific requirements
    print("\n" + "="*85)
    print(f" 🎯 COMPARATIVE ANALYSIS FINAL RESULTS ({args.mc_reps} REPETITIONS)")
    print("="*85)
    print(f"{'Estimator':<12} | {'Mean V(d_hat)':<18} | {'Std V(d_hat)':<18} | {'Est. Error |V-V_hat|':<22}")
    print("-" * 85)
    for row in summary_data:
        print(f"{row['Estimator']:<12} | {row['Mean_True_Q_V(d)']:<18.6f} | {row['Std_True_Q_V(d)']:<18.6f} | {row['Estimation_Error_Mean']:<22.6f}")
    print("="*85 + "\n")
    
    # 追加统计结果到同一个文件的末尾
    with open(save_path, 'a') as f:
        f.write("\n=== SUMMARY STATISTICS ===\n")
    df_summary.to_csv(save_path, mode='a', index=False)
    
    print(f"💾 Report safely saved correctly to: {save_path}")

    # Generate and save boxplot
    plot_boxplot(results, args, res_dir)

if __name__ == "__main__":
    args = parse_arguments()
    run_comparative_mc(args)
