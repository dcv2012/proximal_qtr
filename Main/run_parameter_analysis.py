import os
import json
import numpy as np
import pandas as pd
from Main.run_mc_simulation import run_monte_carlo

def run_parameter_grid_analysis():
    # 模拟设置
    n_train_list = [500, 1000, 5000, 10000]
    phi_types = [1, 2, 3, 4]
    model_types = ["linear", "nn"]
    tau = 0.5
    seed = 20026
    mc_reps = 100
    
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
                
                # 运行 MC 仿真 (mc_n 内部自动计算为 10*n_train)
                q_mean, q_std, q_values = run_monte_carlo(
                    n_train=n_train, seed=seed, tau=tau, 
                    phi_type=p_type, model_type=m_type, 
                    mc_reps=mc_reps, compare_oracle=True
                )
                
                # 为了获取 oracle 收益，重新采样一个极大样本集来逼近真值
                from Main.src.data_generate import intervened_data_gen, adjust_para_set_for_new_coding, origin_para_set
                params = adjust_para_set_for_new_coding(origin_para_set)
                df_oracle = intervened_data_gen(n_train * 50, params, a=[1, 1])
                oracle_q = np.quantile(df_oracle['Y2'], tau)
                
                # 计算统计指标
                regret = oracle_q - q_mean
                rmse = np.sqrt(np.mean((oracle_q - np.array(q_values))**2))
                
                # 记录结果
                res_dict = {
                    "n_train": n_train,
                    "tau": tau,
                    "phi_type": p_type,
                    "model_type": m_type,
                    "oracle_q": float(oracle_q),
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
    run_parameter_grid_analysis()
