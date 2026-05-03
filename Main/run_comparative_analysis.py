import numpy as np
import pandas as pd
import argparse
import os
import time
import torch
import random
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from Main.src.data_generate import dynamic_intervened_data_gen, origin_para_set
from Main.src.prox_qtr_sl.estimate_prox_qtr_sl import train_policy_prox_qtr_sl
from Main.src.SRA.estimate_SRA import train_policy_SRA, train_policy_SRA_no_cf
from Main.src.Oracle.estimate_Oracle import train_policy_Oracle, train_policy_Oracle_no_cf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIRNAME = "comparative_analysis_502"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Monte Carlo Comparative Analysis for Proximal QTR, SRA, and Oracle")
    
    # 手动可调数据集参数/调参设定
    parser.add_argument("--n_train", type=int, default=2000, help="Training set sample size (e.g. 500, 1000, 2000, 5000)")
    parser.add_argument("--mc_reps", type=int, default=30, help="Number of Monte Carlo repetitions")
    parser.add_argument("--mc_eval_size", type=int, default=100000, help="Size of data generated for policy evaluation")
    parser.add_argument("--seed", type=int, default=285063, help="Base random seed for data generation and model initializations")
    
    # 策略模型参数
    parser.add_argument("--tau", type=float, default=0.5, help="Target quantile level (e.g., 0.5 for median)")
    parser.add_argument("--phi_type", type=int, choices=[0, 1, 2, 3, 4], default=1, help="Type of surrogate loss function")
    parser.add_argument("--model_type", type=str, choices=["linear", "nn"], default="nn", help="Class of policy function U_n")
    parser.add_argument("--k_folds", type=int, default=2, help="Number of folds for cross-fitting")
    parser.add_argument("--max_alt_iters", type=int, default=20, help="Max iterations for SCL optimization")
    parser.add_argument("--dgp", type=str, choices=["S1", "S2"], default="S1", help="Outcome scenario with S1-linear, S2-nonlinear")
    parser.add_argument("--optim_mode", type=str, choices=["scl", "ao"], default="ao", help="Optimization framework for SRA/Oracle (scl=Binary Search, ao=Grid Search)")
    parser.add_argument("--no_cf", action="store_true", help="Skip cross-fitting for SRA and Oracle (faster)")
    parser.add_argument("--mmr_loss", type=str, choices=["U_statistic", "V_statistic"], default="V_statistic", help="MMR loss formulation for treatment bridge estimation")
    parser.add_argument("--q22_output_bound", type=float, default=4, help="Symmetric tanh output bound C for q22 bridge estimates")
    
    return parser.parse_args()


def plot_boxplot_from_df(df_valid, args, res_dir):
    # Prepare data for plotting
    plt.figure(figsize=(8, 6))
    
    # df_valid contains 'Estimator', 'True_Perf', 'Est_Error'
    sns.boxplot(x="Estimator", y="True_Perf", data=df_valid, palette="Set2")
    plt.title(f"True Quantile Value of Estimated Policies\n(n={args.n_train}, tau={args.tau}, reps={args.mc_reps}, phi_type={args.phi_type})")
    plt.ylabel(r"Target Quantile Value $V(\hat{d})$")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    fname = f"boxplot_{args.dgp}_n{args.n_train}_phi{args.phi_type}_{args.model_type}__C{args.q22_output_bound}_tau{args.tau}_reps{args.mc_reps}.png"
    save_path = os.path.join(res_dir, fname)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Boxplot successfully saved to: {save_path}")
        

def run_comparative_mc(args):
    print("\n" + "#"*70)
    print("   MONTE CARLO COMPARATIVE ANALYSIS (Proximal vs SRA vs Oracle)")
    print("#"*70)
    print(f"Configurations:")
    for k, v in vars(args).items():
        print(f" - {k}: {v}")
    print(f" - Device: {device}\n")

    
    params = origin_para_set
    mc_sample_size = args.mc_eval_size
    
    results = {
        "Proximal": {"true_perf": [], "est_error": []},
        "SRA": {"true_perf": [], "est_error": []},
        "Oracle": {"true_perf": [], "est_error": []}
    }
    
    # 提前定义好保存路径并在开始前写入由于随时追踪原始结果的文件头
    res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', RESULTS_DIRNAME)
    os.makedirs(res_dir, exist_ok=True)
    fname = f"raw_{args.dgp}_n{args.n_train}_phi{args.phi_type}_{args.model_type}_C{args.q22_output_bound}_tau{args.tau}_reps{args.mc_reps}.csv"
    save_path = os.path.join(res_dir, fname)
    
    # 注意：为了让 analyze_results 可以读取无噪音的数据格式，我们纯粹记录数据点
    with open(save_path, 'w') as f:
        f.write("Repetition,Estimator,True_Perf,Est_Error\n")
    
    st_time = time.time()
    
    for i in range(args.mc_reps):
        current_seed = args.seed + i
        eval_seed = current_seed + 99999
        print(f"\n[{i+1}/{args.mc_reps}] MC Repetition Seed: {current_seed}")
        
        # === Enforce Absolute Determinism ===
        random.seed(current_seed)
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        

        # === 1. Proximal QTR ===
        print(f"  -> Training Proximal QTR (Always Cross-Fitting)")
        f1_p, f2_p, q_est_p, _ = train_policy_prox_qtr_sl(
            n_train=args.n_train, seed=current_seed, K_folds=args.k_folds, 
            max_alt_iters=args.max_alt_iters, tau=args.tau, 
            phi_type=args.phi_type, model_type=args.model_type, save_models=False,
            dgp=args.dgp, mmr_loss=args.mmr_loss, q22_output_bound=args.q22_output_bound
        )
        
        df_eval_p = dynamic_intervened_data_gen(mc_sample_size, params, f1=f1_p, f2=f2_p, device=device, seed=eval_seed, scenario=args.dgp)
        true_perf_p = np.quantile(df_eval_p['Y2'], args.tau)
        results["Proximal"]["true_perf"].append(true_perf_p)
        results["Proximal"]["est_error"].append(true_perf_p - q_est_p)
        
        # === 2. SRA Estimator ===
        print(f"  -> Training SRA Estimator (No-CF: {args.no_cf})...")
        if args.no_cf:
            f1_s, f2_s, q_est_s, _ = train_policy_SRA_no_cf(
                n_train=args.n_train, seed=current_seed,
                max_alt_iters=args.max_alt_iters, tau=args.tau, 
                phi_type=args.phi_type, model_type=args.model_type, save_models=False,
                dgp=args.dgp, optim_mode=args.optim_mode
            )
        else:
            f1_s, f2_s, q_est_s, _ = train_policy_SRA(
                n_train=args.n_train, seed=current_seed, K_folds=args.k_folds,
                max_alt_iters=args.max_alt_iters, tau=args.tau, 
                phi_type=args.phi_type, model_type=args.model_type, save_models=False,
                dgp=args.dgp, optim_mode=args.optim_mode
            )
        df_eval_s = dynamic_intervened_data_gen(mc_sample_size, params, f1=f1_s, f2=f2_s, device=device, seed=eval_seed, scenario=args.dgp)
        true_perf_s = np.quantile(df_eval_s['Y2'], args.tau)
        results["SRA"]["true_perf"].append(true_perf_s)
        results["SRA"]["est_error"].append(true_perf_s - q_est_s)
        
        # === 3. Oracle Estimator ===
        print(f"  -> Training Oracle Estimator (No-CF: {args.no_cf})...")
        if args.no_cf:
            f1_o, f2_o, q_est_o, _ = train_policy_Oracle_no_cf(
                n_train=args.n_train, seed=current_seed,
                max_alt_iters=args.max_alt_iters, tau=args.tau, 
                phi_type=args.phi_type, model_type=args.model_type, save_models=False,
                dgp=args.dgp, optim_mode=args.optim_mode
            )
        else:
            f1_o, f2_o, q_est_o, _ = train_policy_Oracle(
                n_train=args.n_train, seed=current_seed, K_folds=args.k_folds,
                max_alt_iters=args.max_alt_iters, tau=args.tau, 
                phi_type=args.phi_type, model_type=args.model_type, save_models=False,
                dgp=args.dgp, optim_mode=args.optim_mode
            )
        df_eval_o = dynamic_intervened_data_gen(mc_sample_size, params, f1=f1_o, f2=f2_o, device=device, seed=eval_seed, scenario=args.dgp)
        true_perf_o = np.quantile(df_eval_o['Y2'], args.tau)
        results["Oracle"]["true_perf"].append(true_perf_o)
        results["Oracle"]["est_error"].append(true_perf_o - q_est_o)
        
        # Debug Report Output for the repetition
        iter_log = (
            f"   Perf  -> Prox: {results['Proximal']['true_perf'][-1]:.12f} | "
            f"SRA: {results['SRA']['true_perf'][-1]:.12f} | "
            f"Oracle: {results['Oracle']['true_perf'][-1]:.12f}"
        )
        print(iter_log)
        
        # 每次循环结束后随时将结果追加记录到文件
        with open(save_path, 'a') as f:
            f.write(f"{i+1},Proximal,{results['Proximal']['true_perf'][-1]:.12f},{results['Proximal']['est_error'][-1]:.12f}\n")
            f.write(f"{i+1},SRA,{results['SRA']['true_perf'][-1]:.12f},{results['SRA']['est_error'][-1]:.12f}\n")
            f.write(f"{i+1},Oracle,{results['Oracle']['true_perf'][-1]:.12f},{results['Oracle']['est_error'][-1]:.12f}\n")
            f.write("\n") # 留一行空格
        
    print(f"\n[Simulation Completed] Execution Time: {(time.time() - st_time)/60:.2f} minutes")
    print(f"💾 Raw results securely saved to: {save_path}")
    return save_path


def analyze_results(csv_path, args, res_dir):
    """
    独立于 MC 采样的结果分析函数，纯粹依赖 csv 记录的内容绘制箱线图并打印计算统计指标。
    """
    if not os.path.exists(csv_path):
        print(f"❌ Error: The specified CSV file was not found at {csv_path}")
        return
        
    print("\n" + "="*85)
    print(" 🎯 ANALYZING COMPARATIVE RESULTS FROM CSV")
    print("="*85)
    
    # 只抽取表头为标准格式的行，防止旧格式残留（如包含SUMMARY的文本）且跳过空格行
    try:
        df_all = pd.read_csv(csv_path, skip_blank_lines=True)
        # Drop rows where True_Perf is text (like the Old Summary headers if user reused a file)
        df_all['True_Perf'] = pd.to_numeric(df_all['True_Perf'], errors='coerce')
        df_all['Est_Error'] = pd.to_numeric(df_all['Est_Error'], errors='coerce')
        df_valid = df_all.dropna(subset=['True_Perf', 'Est_Error'])
    except Exception as e:
         print(f"❌ Error reading the CSV file: {e}")
         return
         
    summary_data = []
    
    for method in ["Proximal", "SRA", "Oracle"]:
        method_data = df_valid[df_valid['Estimator'] == method]
        if len(method_data) == 0:
            continue
            
        perf_array = method_data['True_Perf'].values
        err_array = method_data['Est_Error'].values
        
        # 1. Mean of true quantile value estimated by the model 
        mean_v_true = np.mean(perf_array)
        
        # 2. Standard deviation of above mean
        std_v_true = np.std(perf_array)
        
        # 3. Mean absolute estimation error (Generalization bound): |V(d_hat) - V_hat(d_hat)|
        estimation_error = np.mean(np.abs(err_array))
        
        summary_data.append({
            "Estimator": method,
            "Mean_True_Q_V(d)": mean_v_true,
            "Std_True_Q_V(d)": std_v_true,
            "Estimation_Error_Mean": estimation_error
        })
        
    if not summary_data:
         print("⚠️ No valid data found in the CSV to summarize.")
         return
         
    df_summary = pd.DataFrame(summary_data)
    
    print(f"{'Estimator':<12} | {'Mean V(d_hat)':<18} | {'Std V(d_hat)':<18} | {'Est. Error |V-V_hat|':<22}")
    print("-" * 85)
    for row in summary_data:
        print(f"{row['Estimator']:<12} | {row['Mean_True_Q_V(d)']:<18.12f} | {row['Std_True_Q_V(d)']:<18.12f} | {row['Estimation_Error_Mean']:<22.12f}")
    print("="*85 + "\n")
    
    summary_path = csv_path.replace('.csv', '_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"💾 Summary statistics saved distinctly to: {summary_path}")

    # 生成箱线图
    plot_boxplot_from_df(df_valid, args, res_dir)
    
if __name__ == "__main__":
    args = parse_arguments()
    
    # 1. 运行并生成数据记录 (如需重新跑实验请取消注释)
    csv_path = run_comparative_mc(args)
    
    '''
    # 2. 给定的结果储存的路径 （如果进行mc实验，则需要注释掉；仅在实验分析时启用）
    res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', RESULTS_DIRNAME)
    fname = f"raw_{args.dgp}_n{args.n_train}_phi{args.phi_type}_{args.model_type}_tau{args.tau}_reps{args.mc_reps}.csv"
    csv_path = os.path.join(res_dir, fname)
    analyze_results(csv_path, args, res_dir)
    '''
    
    
    
    
