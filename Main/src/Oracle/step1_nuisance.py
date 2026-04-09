import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any, Tuple, Callable

def estimate_nuisance(df_train: pd.DataFrame, df_val: pd.DataFrame, n_trials: int = 1) -> Tuple[Callable[[pd.DataFrame], np.ndarray], Dict[str, Any]]:
    """
    Oracle 估计器的 Step 1: 使用 Logistic Regression 估计倾向得分。
    Oracle 具有访问真实未观测混杂 U 的特权，并按照 DGP 的真实函数形式 (Logistic) 进行建模。
    """
    # 按照 method.tex 4.3.3 定义的 X1_Oracle 和 X2_Oracle
    # 这也完全符合 DGP (426, 437) 的输入变量
    features1 = ['Y0', 'U0']
    features2 = ['Y0', 'Y1', 'U0', 'U1', 'A1']
    
    y1_train = (df_train['A1'] == 1).astype(int)
    y2_train = (df_train['A2'] == 1).astype(int)
    
    print("Training Oracle Propensity Models (Logistic Regression - DGPSatisfied)...")
    clf1 = LogisticRegression(penalty=None, solver='lbfgs') # Oracle 理论上不应需要 L2 惩罚以保持无偏性
    clf1.fit(df_train[features1], y1_train)
    
    clf2 = LogisticRegression(penalty=None, solver='lbfgs')
    clf2.fit(df_train[features2], y2_train)
    
    def predict_weights(df_test: pd.DataFrame) -> np.ndarray:
        prob1 = clf1.predict_proba(df_test[features1])[:, 1]
        prob2 = clf2.predict_proba(df_test[features2])[:, 1]
        
        A1 = df_test['A1'].values
        A2 = df_test['A2'].values
        
        pi1 = np.where(A1 == 1, prob1, 1 - prob1)
        pi2 = np.where(A2 == 1, prob2, 1 - prob2)
        
        # 数值稳定性
        pi1 = np.clip(pi1, 1e-6, 1.0)
        pi2 = np.clip(pi2, 1e-6, 1.0)
        
        return 1.0 / (pi1 * pi2)
        
    return predict_weights, {"clf1": clf1.get_params(), "clf2": clf2.get_params()}
