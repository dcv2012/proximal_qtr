import os
import time
import pandas as pd
import itertools

from Main.Experiment import train_policy

def main():
    # 论文中 4.3 的网格搜索参数要求
    n_train_list = [500, 2000, 5000]
    tau_list = [0.5]
    phi_type_list = [1, 2, 3, 4]
    model_type_list = ["linear", "nn"]
    seed = 2026
    
    # 结果保存路径
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 组合所有的测试环境
    configs = list(itertools.product(n_train_list, tau_list, phi_type_list, model_type_list))
    total_configs = len(configs)
    
    print(f"🚀 Starting Performance Analysis for {total_configs} configurations...")
    
    # 整体汇总 DataFrame
    master_df = pd.DataFrame()
    
    for i, (n_train, tau, phi_type, model_type) in enumerate(configs):
        print(f"\n=======================================================")
        print(f"[{i+1}/{total_configs}] Running: n_train={n_train}, tau={tau}, phi_type={phi_type}, model_type={model_type}")
        print(f"=======================================================\n")
        
        # 记录开始时间作为 Runtime 计算的依据
        start_time = time.time()
        
        # 调用核心入口（不需要保存 pt 模型以防撑爆硬盘）
        f1, f2, q_opt, sv_opt = train_policy(
            n_train=n_train, 
            seed=seed, 
            K_folds=2, 
            max_alt_iters=3, 
            tau=tau, 
            phi_type=phi_type, 
            model_type=model_type, 
            save_models=False
        )
        
        runtime = time.time() - start_time
        
        record = {
            'n_train': n_train,
            'tau': tau,
            'phi_type': phi_type,
            'model_type': model_type,
            'seed': seed,
            'runtime_seconds': runtime,
            'estimated_q': q_opt,
            'sv_psi': sv_opt
        }
        
        # 1. 单独按参数命名保存
        config_name = f"ntrain{n_train}_tau{tau}_phi{phi_type}_model{model_type}_seed{seed}"
        result_path = os.path.join(results_dir, f"result_{config_name}.csv")
        df_single = pd.DataFrame([record])
        df_single.to_csv(result_path, index=False)
        print(f"\n📁 Saved iteration details to -> {result_path}")
        
    print(f"\n✅ All {total_configs} experiments completed successfully!")

if __name__ == "__main__":
    main()
