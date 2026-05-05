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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESULTS_DIRNAME = "prox_505"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Monte Carlo Analysis for Proximal QTR Only")

    # 数据集参数
    parser.add_argument("--n_train", type=int, default=2000, help="Training set sample size (e.g. 500, 1000, 2000, 5000)")
    parser.add_argument("--mc_reps", type=int, default=30, help="Number of Monte Carlo repetitions")
    parser.add_argument("--mc_eval_size", type=int, default=100000, help="Size of data generated for policy evaluation")
    parser.add_argument("--seed", type=int, default=285063, help="Base random seed for data generation and model initializations")

    # 策略模型参数
    parser.add_argument("--tau", type=float, default=0.5, help="Target quantile level (e.g., 0.5 for median)")
    parser.add_argument("--phi_type", type=int, choices=[0, 1, 2, 3, 4], default=1, help="Type of surrogate loss function")
    parser.add_argument("--model_type", type=str, choices=["linear", "nn"], default="nn", help="Class of policy function U_n")
    parser.add_argument("--k_folds", type=int, default=2, help="Number of folds for cross-fitting")
    parser.add_argument("--max_alt_iters", type=int, default=10, help="Max iterations for alternating optimization")
    parser.add_argument("--dgp", type=str, choices=["S1", "S2"], default="S1", help="Outcome scenario with S1-linear, S2-nonlinear")
    parser.add_argument("--mmr_loss", type=str, choices=["U_statistic", "V_statistic"], default="V_statistic", help="MMR loss formulation for treatment bridge estimation")
    parser.add_argument("--q22_output_bound", type=float, default=4, help="Symmetric tanh output bound C for q22 bridge estimates")

    return parser.parse_args()


def plot_boxplot_from_df(df_valid, args, res_dir):
    plt.figure(figsize=(6, 5))

    # df_valid contains 'Estimator', 'True_Perf', 'Est_Error'
    sns.boxplot(x="Estimator", y="True_Perf", data=df_valid, palette="Set2")
    plt.title(f"True Quantile Value — Proximal QTR\n(n={args.n_train}, tau={args.tau}, reps={args.mc_reps}, phi_type={args.phi_type})")
    plt.ylabel(r"Target Quantile Value $QV(\tau, \hat{\bar{d}}_2)$")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    fname = f"boxplot_prox_{args.dgp}_n{args.n_train}_phi{args.phi_type}_{args.model_type}_C{args.q22_output_bound}_tau{args.tau}_reps{args.mc_reps}.png"
    save_path = os.path.join(res_dir, fname)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Boxplot saved to: {save_path}")


def run_prox_mc(args):
    print("\n" + "#"*70)
    print("   MONTE CARLO ANALYSIS — PROXIMAL QTR ONLY")
    print("#"*70)
    print("Configurations:")
    for k, v in vars(args).items():
        print(f" - {k}: {v}")
    print(f" - Device: {device}\n")

    params = origin_para_set
    mc_sample_size = args.mc_eval_size

    results = {
        "Proximal": {"true_perf": [], "est_error": []}
    }

    # 保存路径
    res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', RESULTS_DIRNAME)
    os.makedirs(res_dir, exist_ok=True)
    fname = f"raw_prox_{args.dgp}_n{args.n_train}_phi{args.phi_type}_{args.model_type}_C{args.q22_output_bound}_tau{args.tau}_reps{args.mc_reps}.csv"
    save_path = os.path.join(res_dir, fname)

    # 写入 CSV 文件头（与 run_comparative_analysis.py 格式完全一致）
    with open(save_path, 'w') as f:
        f.write("Repetition,Estimator,True_Perf,Est_Error\n")

    st_time = time.time()

    for i in range(args.mc_reps):
        current_seed = args.seed + i
        eval_seed = current_seed + 99999
        print(f"\n[{i+1}/{args.mc_reps}] MC Repetition Seed: {current_seed}")

        # === 强制确定性 ===
        random.seed(current_seed)
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # === Proximal QTR ===
        print(f"  -> Training Proximal QTR (Always Cross-Fitting)")
        f1_p, f2_p, q_est_p, _ = train_policy_prox_qtr_sl(
            n_train=args.n_train, seed=current_seed, K_folds=args.k_folds,
            max_alt_iters=args.max_alt_iters, tau=args.tau,
            phi_type=args.phi_type, model_type=args.model_type, save_models=False,
            dgp=args.dgp, mmr_loss=args.mmr_loss, q22_output_bound=args.q22_output_bound
        )

        df_eval_p = dynamic_intervened_data_gen(mc_sample_size, params, f1=f1_p, f2=f2_p, device=device, seed=eval_seed, scenario=args.dgp)
        true_perf_p = np.quantile(df_eval_p['Y2'], args.tau)
        est_error_p = true_perf_p - q_est_p

        results["Proximal"]["true_perf"].append(true_perf_p)
        results["Proximal"]["est_error"].append(est_error_p)

        print(f"   Perf -> Prox: {true_perf_p:.12f} | Est_Error: {est_error_p:.12f}")

        # 实时追加到 CSV（格式与 run_comparative_analysis.py 完全一致）
        with open(save_path, 'a') as f:
            f.write(f"{i+1},Proximal,{true_perf_p:.12f},{est_error_p:.12f}\n")
            f.write("\n")  # 保留空行占位

    print(f"\n[Simulation Completed] Execution Time: {(time.time() - st_time)/60:.2f} minutes")
    print(f"💾 Raw results saved to: {save_path}")
    return save_path


def analyze_results(csv_path, args, res_dir):
    """
    从 CSV 文件中读取数据，计算统计指标并绘制箱线图。
    与 run_comparative_analysis.py 中的 analyze_results 保持完全一致的逻辑。
    """
    if not os.path.exists(csv_path):
        print(f"❌ Error: The specified CSV file was not found at {csv_path}")
        return

    print("\n" + "="*85)
    print(" 🎯 ANALYZING PROXIMAL QTR RESULTS FROM CSV")
    print("="*85)

    try:
        df_all = pd.read_csv(csv_path, skip_blank_lines=True)
        df_all['True_Perf'] = pd.to_numeric(df_all['True_Perf'], errors='coerce')
        df_all['Est_Error'] = pd.to_numeric(df_all['Est_Error'], errors='coerce')
        df_valid = df_all.dropna(subset=['True_Perf', 'Est_Error'])
    except Exception as e:
        print(f"❌ Error reading the CSV file: {e}")
        return

    summary_data = []

    for method in ["Proximal"]:
        method_data = df_valid[df_valid['Estimator'] == method]
        if len(method_data) == 0:
            continue

        perf_array = method_data['True_Perf'].values
        err_array = method_data['Est_Error'].values

        # 1. 真实分位数值的均值
        mean_v_true = np.mean(perf_array)

        # 2. 真实分位数值的标准差
        std_v_true = np.std(perf_array)

        # 3. 平均绝对估计误差 |QV(d_hat) - QV_hat(d_hat)|
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
    print(f"💾 Summary statistics saved to: {summary_path}")

    # 生成箱线图
    plot_boxplot_from_df(df_valid, args, res_dir)


if __name__ == "__main__":
    args = parse_arguments()

    # 1. 运行蒙特卡洛实验并记录数据
    csv_path = run_prox_mc(args)

    # 2. 分析实验结果（若仅需重跑分析，注释掉上方 run_prox_mc，并手动指定 csv_path）
    res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', RESULTS_DIRNAME)
    analyze_results(csv_path, args, res_dir)
