import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(results_dir):
    all_files = glob.glob(os.path.join(results_dir, "result_*.csv"))
    if not all_files:
        print(f"No result CSV files found in {results_dir}")
        return None
        
    df_list = [pd.read_csv(f) for f in all_files]
    return pd.concat(df_list, ignore_index=True)

def plot_performance():
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    df = load_results(results_dir)
    
    if df is None:
        print("Please run `run_performance_analysis.py` first to generate the data.")
        return
        
    # 使用 seaborn 全局样式提升排版质感
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    plot_dir = os.path.join(results_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. 运行时效分析 (Runtime Analysis)
    # y=runtime, x=n_train, hue=model_type
    plt.figure(figsize=(9, 6))
    
    # 用 Pointplot 或 Lineplot 画出时间随数据量上升的斜率折线
    sns.pointplot(data=df, x='n_train', y='runtime_seconds', hue='model_type', 
                  markers=['o', 's'], linestyles=['-', '--'], errorbar=('ci', 95))
    
    plt.title('Algorithm Runtime Efficiency vs Sample Size', fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('Wall-clock Runtime (Seconds)', fontsize=13)
    plt.xlabel('Training Set Size ($N$)', fontsize=13)
    plt.legend(title='Policy Dimension Type ($\mathcal{U}_1, \mathcal{U}_2$)')
    
    # 保存并防修剪
    save_path_1 = os.path.join(plot_dir, "runtime_analysis.png")
    plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 估计器极值分析 (Estimator \eqref{est} value / Q-value)
    # y=estimated_q, x=phi_type, hue=model_type (按照具体的 n_train 隔离画图，或者在一张图用 facet)
    
    unique_n_trains = sorted(df['n_train'].unique())
    
    for sample_size in unique_n_trains:
        sub_df = df[df['n_train'] == sample_size]
        if sub_df.empty:
            continue
            
        plt.figure(figsize=(9, 6))
        
        # 柱状图对比四种不同平滑损失函数的极值逼近效果
        ax = sns.barplot(data=sub_df, x='phi_type', y='estimated_q', hue='model_type',
                         palette='rocket')
                         
        plt.title(f'Converged Quantile Estimator Value across Surrogates ($N={sample_size}$)', 
                  fontsize=14, fontweight='bold', pad=15)
        plt.ylabel('Estimated Target Value ($\hat{QV}(\tau)$)', fontsize=13)
        plt.xlabel('Surrogate $\phi$ Loss Form', fontsize=13)
        plt.legend(title='Policy Network')
        
        # 基于分位数实际大小动态限制Y轴，避免图形被0吃穿从而拉开对比差距
        min_q = sub_df['estimated_q'].min()
        max_q = sub_df['estimated_q'].max()
        padding = (max_q - min_q) * 0.2 if max_q != min_q else max_q * 0.05
        plt.ylim(min_q - padding, max_q + padding)
        
        save_path_2 = os.path.join(plot_dir, f"estimator_comparison_n{sample_size}.png")
        plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"✅ Awesome! Scientific plots are rendered and saved at: [{plot_dir}]")

if __name__ == "__main__":
    plot_performance()
