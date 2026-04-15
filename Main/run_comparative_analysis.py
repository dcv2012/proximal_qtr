import numpy as np
import pandas as pd
import argparse
import os
import time
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from Main.src.prox_qtr_sl.estimate_prox_qtr_sl import train_policy_prox_qtr_sl, train_policy_prox_qtr_no_cf
from Main.src.SRA.estimate_SRA import train_policy_SRA, train_policy_SRA_no_cf
from Main.src.Oracle.estimate_Oracle import train_policy_Oracle, train_policy_Oracle_no_cf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_arguments():
    parser = argparse.ArgumentParser(description="Monte Carlo Comparative Analysis for Proximal QTR, SRA, and Oracle")
    
    # 手动可调数据集参数/调参设定
    parser.add_argument("--n_train", type=int, default=1000, help="Training set sample size (e.g. 500, 1000, 2000, 5000)")
    parser.add_argument("--mc_reps", type=int, default=30, help="Number of Monte Carlo repetitions")
    parser.add_argument("--mc_eval_size", type=int, default=100000, help="Size of data generated for policy evaluation")
    parser.add_argument("--seed", type=int, default=20026, help="Base random seed for data generation and model initializations")
    
    # 策略模型参数
    parser.add_argument("--tau", type=float, default=0.5, help="Target quantile level (e.g., 0.5 for median)")
    parser.add_argument("--phi_type", type=int, choices=[0, 1, 2, 3, 4], default=1, help="Type of surrogate loss function")
    parser.add_argument("--model_type", type=str, choices=["linear", "nn"], default="linear", help="Class of policy function U_n")
    parser.add_argument("--k_folds", type=int, default=2, help="Number of folds for cross-fitting")
    parser.add_argument("--max_alt_iters", type=int, default=20, help="Max iterations for SCL optimization")
    parser.add_argument("--dgp", type=str, choices=["S1", "S2"], default="S2", help="Data generation process version to use (S1=data_generate, S2=data_generate_new)")
    parser.add_argument("--no_cf", action="store_true", help="Skip cross-fitting for Proximal QTR (faster)")
    
    return parser.parse_args()


def plot_boxplot_from_df(df_valid, args, res_dir):
    # Prepare data for plotting
    plt.figure(figsize=(8, 6))
    
    # df_valid contains 'Estimator', 'True_Perf', 'Train_Est'
    sns.boxplot(x="Estimator", y="True_Perf", data=df_valid, palette="Set2")
    plt.title(f"True Quantile Value of Estimated Policies\n(n={args.n_train}, tau={args.tau}, reps={args.mc_reps}, phi_type={args.phi_type})")
    plt.ylabel(r"Target Quantile Value $V(\hat{d})$")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    fname = f"boxplot_n{args.n_train}_tau{args.tau}_phi{args.phi_type}_{args.model_type}_reps{args.mc_reps}.png"
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
    
    
    if args.dgp == "S1":
        from Main.src.data_generate import dynamic_intervened_data_gen, adjust_para_set_for_new_coding, origin_para_set
    else:
        from Main.src.data_generate_new import dynamic_intervened_data_gen, adjust_para_set_for_new_coding, origin_para_set
    
    params = adjust_para_set_for_new_coding(origin_para_set)
    mc_sample_size = args.mc_eval_size
    
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
    
    # 注意：为了让 analyze_results 可以读取无噪音的数据格式，我们纯粹记录数据点
    with open(save_path, 'w') as f:
        f.write("Repetition,Estimator,True_Perf,Train_Est\n")
    
    st_time = time.time()
    
    for i in range(args.mc_reps):
        current_seed = args.seed + i
        print(f"\n[{i+1}/{args.mc_reps}] MC Repetition Seed: {current_seed}")
        
        # === 1. Proximal QTR ===
        print(f"  -> Training Proximal QTR (No-CF: {args.no_cf})...")
        if args.no_cf:
            f1_p, f2_p, q_est_p, _ = train_policy_prox_qtr_sl(
                n_train=args.n_train, seed=current_seed, 
                max_alt_iters=args.max_alt_iters, tau=args.tau, 
                phi_type=args.phi_type, model_type=args.model_type, save_models=False,
                dgp=args.dgp
            )
        else:
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
        print(f"  -> Training SRA Estimator (No-CF: {args.no_cf})...")
        if args.no_cf:
            f1_s, f2_s, q_est_s, _ = train_policy_SRA_no_cf(
                n_train=args.n_train, seed=current_seed,
                max_alt_iters=args.max_alt_iters, tau=args.tau, 
                phi_type=args.phi_type, model_type=args.model_type, save_models=False,
                dgp=args.dgp
            )
        else:
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
        print(f"  -> Training Oracle Estimator (No-CF: {args.no_cf})...")
        if args.no_cf:
            f1_o, f2_o, q_est_o, _ = train_policy_Oracle_no_cf(
                n_train=args.n_train, seed=current_seed,
                max_alt_iters=args.max_alt_iters, tau=args.tau, 
                phi_type=args.phi_type, model_type=args.model_type, save_models=False,
                dgp=args.dgp
            )
        else:
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
    
    # 只抽取表头为标准格式的行，防止旧格式残留（如包含SUMMARY的文本）
    try:
        df_all = pd.read_csv(csv_path)
        # Drop rows where True_Perf is text (like the Old Summary headers if user reused a file)
        df_all['True_Perf'] = pd.to_numeric(df_all['True_Perf'], errors='coerce')
        df_all['Train_Est'] = pd.to_numeric(df_all['Train_Est'], errors='coerce')
        df_valid = df_all.dropna(subset=['True_Perf', 'Train_Est'])
    except Exception as e:
         print(f"❌ Error reading the CSV file: {e}")
         return
         
    summary_data = []
    
    for method in ["Proximal", "SRA", "Oracle"]:
        method_data = df_valid[df_valid['Estimator'] == method]
        if len(method_data) == 0:
            continue
            
        perf_array = method_data['True_Perf'].values
        est_array = method_data['Train_Est'].values
        
        # 1. Mean of true quantile value estimated by the model 
        mean_v_true = np.mean(perf_array)
        
        # 2. Standard deviation of above mean
        std_v_true = np.std(perf_array)
        
        # 3. Mean absolute estimation error (Generalization bound): |V(d_hat) - V_hat(d_hat)|
        estimation_error = np.mean(np.abs(perf_array - est_array))
        
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
        print(f"{row['Estimator']:<12} | {row['Mean_True_Q_V(d)']:<18.6f} | {row['Std_True_Q_V(d)']:<18.6f} | {row['Estimation_Error_Mean']:<22.6f}")
    print("="*85 + "\n")
    
    summary_path = csv_path.replace('.csv', '_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"💾 Summary statistics saved distinctly to: {summary_path}")

    # 生成箱线图
    plot_boxplot_from_df(df_valid, args, res_dir)

if __name__ == "__main__":
    args = parse_arguments()
    
    # 1. 运行并生成数据记录 (已注释，如需重新跑实验请取消注释)
    # csv_path = run_comparative_mc(args)
    
    # 2. 给定的结果储存的路径
    res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'comparative_analysis')
    fname = f"comparative_n{args.n_train}_tau{args.tau}_phi{args.phi_type}_{args.model_type}_{args.dgp}_reps{args.mc_reps}.csv"
    csv_path = os.path.join(res_dir, fname)
    
    # 从数据记录中分析和画图
    analyze_results(csv_path, args, res_dir)
