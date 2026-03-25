import os
import json, csv
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch

from src.data.data_class import (
    MMRTrainDataSet_q,
    MMRTrainDataSetTorch_q,
    MMRTrainDataSet_q_2stage,
    MMRTrainDataSetTorch_q_2stage,
    MMRTestDataSet_q,
    MMRTestDataSetTorch_q,
    MMRTestDataSet_q_2stage,
    MMRTestDataSetTorch_q_2stage,
)
from src.data.simu_new import (
    generate_train_simulation_q,
    generate_test_simulation_q,
    generate_train_simulation_q_2stage,
    generate_test_simulation_q_2stage,
)
from src.models.MMR.MMR_trainers import MMR_Trainer_Simulation, MMR_Trainer_Simulation_2stage
from src.models.MMR.MMR_model import MLP_for_MMR
from src.utils.setseed import set_seed

##--------------

def load_hyperparams(treatment1:int, treatment2:int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # 加载预先搜索的 hyperparam json
    basedir = Path(Path(str(Path(__file__).parent.parent.parent.parent / 'configs' / 'simulation')))
    dir_s1 = basedir / f"deepmmr_a1={treatment1}_1s.json"
    dir_s2 = basedir / f"deepmmr_a1={treatment1}_a2={treatment2}_2s.json"
    
    if not dir_s1.exists() or not dir_s2.exists():
        raise FileNotFoundError(f"hyperparam files not found: {dir_s1}, {dir_s2}")

    model_config_s1 = json.loads(dir_s1.read_text())
    model_config_s2 = json.loads(dir_s2.read_text())
    
    return model_config_s1, model_config_s2


def screen_test_data_2s(test_data_t: MMRTestDataSetTorch_q_2stage, test_mask: torch.Tensor) -> MMRTestDataSetTorch_q_2stage:
     return MMRTestDataSetTorch_q_2stage(
         treatment1=test_data_t.treatment1[test_mask].unsqueeze(-1),
         treatment_proxy1=test_data_t.treatment_proxy1[test_mask].unsqueeze(-1),
         outcome1=test_data_t.outcome1[test_mask].unsqueeze(-1),
         outcome_proxy1=test_data_t.outcome_proxy1[test_mask].unsqueeze(-1),
         backdoor1=test_data_t.backdoor1[test_mask].unsqueeze(-1),
         structural1=test_data_t.structural1,
         treatment2=test_data_t.treatment2[test_mask].unsqueeze(-1),
         treatment_proxy2=test_data_t.treatment_proxy2[test_mask].unsqueeze(-1),
         outcome2=test_data_t.outcome2[test_mask].unsqueeze(-1),
         outcome_proxy2=test_data_t.outcome_proxy2[test_mask].unsqueeze(-1),
         backdoor2=test_data_t.backdoor2[test_mask].unsqueeze(-1)
            )

class MMREstimator_q:
    # 根据a1和a2的值预测q22的模型
    def __init__(self, treatment1: int, treatment2: int, random_seed: int):
        self.a1 = treatment1
        self.a2 = treatment2
        self.model_config_s1, self.model_config_s2 = load_hyperparams(self.a1, self.a2)
        set_seed(random_seed)
        
        self.model_1s = None
        self.model_2s = None
        self.trainer_1s = None
        self.trainer_2s = None
        
    @staticmethod
    def generate_data(n_train: int, n_test: int, scenario: str='s1') -> Tuple[MMRTrainDataSet_q_2stage, MMRTrainDataSet_q_2stage, MMRTestDataSet_q_2stage]:
        train_data = generate_train_simulation_q_2stage(int(n_train * 0.75), scenario)
        val_data = generate_train_simulation_q_2stage(int(n_train * 0.25), scenario)
        test_data = generate_test_simulation_q_2stage(n_test, scenario)
        
        return train_data, val_data, test_data
    
    def train_1s(self, train_data: MMRTrainDataSet_q, val_data: MMRTrainDataSet_q) -> None:
        train_data_matched = MMRTrainDataSet_q(
            treatment=train_data.treatment,
            treatment_target=(train_data.treatment == self.a1),
            treatment_proxy=train_data.treatment_proxy,
            outcome_proxy=train_data.outcome_proxy,
            outcome=train_data.outcome,
            backdoor=train_data.backdoor
        )
        train_data_t = MMRTrainDataSetTorch_q.from_numpy(train_data_matched)
        
        val_data_matched = MMRTrainDataSet_q(
            treatment=val_data.treatment,
            treatment_target=(val_data.treatment == self.a1),
            treatment_proxy=val_data.treatment_proxy,
            outcome_proxy=val_data.outcome_proxy,
            outcome=val_data.outcome,
            backdoor=val_data.backdoor
        )
        val_data_t = MMRTrainDataSetTorch_q.from_numpy(val_data_matched)
        
        trainer_1s = MMR_Trainer_Simulation(self.model_config_s1)
        _, model = trainer_1s.train(train_data_t, val_data_t)
        
        self.trainer_1s = trainer_1s
        self.model_1s = model
    
    def train_2s(self, train_data: MMRTrainDataSet_q_2stage, val_data: MMRTrainDataSet_q_2stage) -> None:
        train_data_matched = MMRTrainDataSet_q_2stage(
            treatment1=train_data.treatment1,
            treatment_target1=(train_data.treatment1 == self.a1),
            treatment_proxy1=train_data.treatment_proxy1,
            outcome_proxy1=train_data.outcome_proxy1,
            outcome1=train_data.outcome1,
            backdoor1=train_data.backdoor1,
            treatment2=train_data.treatment2,
            treatment_target2=(train_data.treatment2 == self.a2),
            treatment_proxy2=train_data.treatment_proxy2,
            outcome2=train_data.outcome2,
            outcome_proxy2=train_data.outcome_proxy2,
            backdoor2=train_data.backdoor2 
        )
        train_data_t = MMRTrainDataSetTorch_q_2stage.from_numpy(train_data_matched)
        
        val_data_matched = MMRTrainDataSet_q_2stage(
                treatment1=val_data.treatment1,
                treatment_target1=(val_data.treatment1 == self.a1),
                treatment_proxy1=val_data.treatment_proxy1,
                outcome_proxy1=val_data.outcome_proxy1,
                outcome1=val_data.outcome1,
                backdoor1=val_data.backdoor1,
                treatment2=val_data.treatment2,
                treatment_target2=(val_data.treatment2 == self.a2),
                treatment_proxy2=val_data.treatment_proxy2,
                outcome2=val_data.outcome2,
                outcome_proxy2=val_data.outcome_proxy2,
                backdoor2=val_data.backdoor2
        )
        val_data_t = MMRTrainDataSetTorch_q_2stage.from_numpy(val_data_matched)
        
        trainer_2s = MMR_Trainer_Simulation_2stage(self.model_config_s1, self.model_config_s2)
        _, [model1, model0] = trainer_2s.train_2s(train_data_t, val_data_t)    # model1--a1=1, model0--a1=-1
        
        self.trainer_2s = trainer_2s
        self.model_2s = (model1 if self.a1==1 else model0)

        
    
    
    def estimate_q11(self, test_data_t: MMRTestDataSetTorch_q) -> np.ndarray:
        q11_val = self.trainer_1s.predict(self.model_1s, test_data_t)
    
    def estimate_q22(self, test_data_t: MMRTestDataSetTorch_q_2stage) -> np.ndarray:
        if not (torch.all(test_data_t.treatment1 == self.a1).item() or torch.all(test_data_t.treatment2 == self.a2).item()):
            raise ValueError("Error: invalid test data")
    
        q22_val = self.trainer_2s.predict_2s(self.model_2s, test_data_t)
        
        return q22_val.cpu().numpy()
        
        




def main_est(seed: int, num_runs: int, n_trains: int, n_tests: int, output_dir: str = './results/simulation'):
    
    set_seed(seed)
    n_experiments = num_runs # 实验次数
    n_train = n_trains  # 训练集大小
    n_test = n_tests   # 测试集大小
    treatment_set = {(1,1),(1,-1),(-1,1),(-1,-1)} 
    
    for i in range(n_experiments): 
        train_data, val_data, test_data = MMREstimator_q.generate_data(n_train, n_test)
        for (a1,a2) in treatment_set:
            # 测试集筛选
            test_data_t = MMRTestDataSetTorch_q_2stage.from_numpy(test_data)
            test_mask = (test_data_t.treatment1 == a1) & (test_data_t.treatment2 == a2)
            test_data_eval_2s = screen_test_data_2s(test_data_t, test_mask)
            
            
            estiamtor = MMREstimator_q(a1, a2, seed)
            estiamtor.train_2s(train_data, val_data)
            q22_numpy = estiamtor.estimate_q22(test_data_eval_2s)

            
            print(f'The test data with (a1={a1}, a2={a2}) for {i+1}th exp has the size {q22_numpy.shape}.')

            
            # 记录数据
            path = Path(output_dir)
            if not path.exists():
                os.makedirs(path)
            file_path = path.joinpath(f"deepmmr_{n_train}_{i+1}th_a1={a1}_a2={a2}.csv")    
                
            with open(file_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                for q22_val in q22_numpy.flatten():
                    writer.writerow([q22_val])
            
                

if __name__ == "__main__":
    main_est(output_dir='./results/simulation')
    
    
    
    