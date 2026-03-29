import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any, Tuple, Callable

def estimate_nuisance(df_train: pd.DataFrame, df_val: pd.DataFrame, n_trials: int = 1) -> Tuple[Callable[[pd.DataFrame], np.ndarray], Dict[str, Any]]:
    """
    SRA 估计器的 Step 1：使用 Logistic Regression 估计倾向得分。
    符合顺序随机化假设 (SRA)，基于观测到的历史变量。
    """
    # 按照 method.tex 4.3.3 定义的 X1_SRA 和 X2_SRA
    features1 = ['Y0', 'Z1', 'W1']
    features2 = ['Y0', 'Y1', 'Z1', 'Z2', 'W1', 'W2', 'A1']
    
    # 合并训练和验证集进行最后模型估计 (Logistic Regression 往往在更多数据上更稳健)
    # 或者只使用训练集。这里遵循 pipeline 习惯使用 train。
    X1_train = df_train[features1]
    y1_train = (df_train['A1'] == 1).astype(int)
    
    X2_train = df_train[features2]
    y2_train = (df_train['A2'] == 1).astype(int)
    
    print("Training SRA Propensity Models (Logistic Regression)...")
    clf1 = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
    clf1.fit(X1_train, y1_train)
    
    clf2 = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
    clf2.fit(X2_train, y2_train)
    
    def predict_weights(df_test: pd.DataFrame) -> np.ndarray:
        # 获取 P(A=1 | X)
        prob1 = clf1.predict_proba(df_test[features1])[:, 1]
        prob2 = clf2.predict_proba(df_test[features2])[:, 1]
        
        # 获取 P(A=a | X)
        A1 = df_test['A1'].values
        A2 = df_test['A2'].values
        
        # A 已经是 {-1, 1}，clf 训练使用的是 {0, 1}
        pi1 = np.where(A1 == 1, prob1, 1 - prob1)
        pi2 = np.where(A2 == 1, prob2, 1 - prob2)
        
        # 数值稳定性截断
        pi1 = np.clip(pi1, 1e-4, 1.0)
        pi2 = np.clip(pi2, 1e-4, 1.0)
        
        return 1.0 / (pi1 * pi2)
        
    return predict_weights, {"clf1": clf1.get_params(), "clf2": clf2.get_params()}
