from src.models.MMR.hp_search import main_hps
from src.models.MMR.MMR_estimator import main_est

def run_exp():
    seed = 2048
    n_train = 2000
    n_test = 1000
    
    n_experiments = 10
    # 搜索超参数
    main_hps(num_runs=n_experiments, n_trains=n_train, n_tests=n_test)
    
    n_experiments = 1
    # 估计q22
    main_est(seed=seed, num_runs = n_experiments, n_trains=n_train, n_tests=n_test)
    
    
    
if __name__ == "__main__":
    run_exp()    