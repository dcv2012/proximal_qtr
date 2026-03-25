import os, pathlib
from pathlib import Path
from typing import Dict, Any

import csv
import time
import json
import numpy as np
import pandas as pd
import torch

from src.data.data_class import MMRTrainDataSet_q, MMRTrainDataSetTorch_q, MMRTestDataSet_q, MMRTestDataSetTorch_q

# from src.data.simulation import generate_test_simulation_h, generate_train_simulation_h
# from src.data.simulation import generate_test_simulation_q, generate_train_simulation_q
from src.data.simu_new import generate_test_simulation_q, generate_train_simulation_q

from src.models.MMR.MMR_trainers import MMR_Trainer_Simulation


def run_mmr_experiments(num_runs: int, scenario: str, kind: str, n_train: int, n_test: int, output_dir: str):
    path = Path(output_dir)
    
    if not path.exists():
        os.makedirs(path)
    file_path = path.joinpath(f"deepmmr_{scenario}_{n_train}.csv")
    
    
    with open(Path(str(pathlib.Path(__file__).parent.parent.parent.parent / 'configs' / 'simulation' / f'mmr_{scenario.lower()}_{kind}_1.json')),'r') as f:
        model_config1 = json.load(f)
    
    with open(Path(str(pathlib.Path(__file__).parent.parent.parent.parent / 'configs' / 'simulation' / f'mmr_{scenario.lower()}_{kind}_0.json')),'r') as f:
        model_config0 = json.load(f)
    
    for i in range(num_runs):
        # fix random seed
        random_seed = np.random.randint(0, 2**20 - 1)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Generate data
        train_data = generate_train_simulation_q(int(n_train * 0.75), scenario)
        val_data = generate_train_simulation_q(int(n_train * 0.25), scenario)
        test_data = generate_test_simulation_q(n_test, scenario) ## 
        

        train_data_1 = MMRTrainDataSet_q(
            treatment=train_data.treatment,
            treatment_target=(train_data.treatment == 1),
            treatment_proxy=train_data.treatment_proxy,
            outcome_proxy=train_data.outcome_proxy,
            outcome=train_data.outcome,
            backdoor=train_data.backdoor
        )
        train_data_1_t = MMRTrainDataSetTorch_q.from_numpy(train_data_1)
        
        val_data_1 = MMRTrainDataSet_q(
            treatment=val_data.treatment,
            treatment_target=(val_data.treatment == 1),
            treatment_proxy=val_data.treatment_proxy,
            outcome_proxy=val_data.outcome_proxy,
            outcome=val_data.outcome,
            backdoor=val_data.backdoor
        )
        val_data_1_t = MMRTrainDataSetTorch_q.from_numpy(val_data_1)
        
        
        train_data_0 = MMRTrainDataSet_q(
            treatment=train_data.treatment,
            treatment_target=(train_data.treatment == -1), #0 -> -1
            treatment_proxy=train_data.treatment_proxy,
            outcome_proxy=train_data.outcome_proxy,
            outcome=train_data.outcome,
            backdoor=train_data.backdoor
        )
        train_data_0_t = MMRTrainDataSetTorch_q.from_numpy(train_data_0)
        
        val_data_0 = MMRTrainDataSet_q(
            treatment=val_data.treatment,
            treatment_target=(val_data.treatment == -1), #0 -> -1
            treatment_proxy=val_data.treatment_proxy,
            outcome_proxy=val_data.outcome_proxy,
            outcome=val_data.outcome,
            backdoor=val_data.backdoor
        )
        val_data_0_t = MMRTrainDataSetTorch_q.from_numpy(val_data_0)
        
        # 测试集分类
        test_data_t = MMRTestDataSetTorch_q.from_numpy(test_data)
        mask1 = (test_data_t.treatment == 1)
        mask0 = (test_data_t.treatment == -1)
        
        test_data_t_1 = MMRTestDataSetTorch_q(
            treatment=test_data_t.treatment[mask1].unsqueeze(-1),
            treatment_proxy=test_data_t.treatment_proxy[mask1].unsqueeze(-1),
            outcome_proxy=test_data_t.outcome_proxy[mask1].unsqueeze(-1),
            outcome=test_data_t.outcome[mask1].unsqueeze(-1),
            backdoor=test_data_t.backdoor[mask1].unsqueeze(-1),
            structural= test_data_t.structural
        )
        test_data_t_0 = MMRTestDataSetTorch_q(
            treatment=test_data_t.treatment[mask0].unsqueeze(-1),
            treatment_proxy=test_data_t.treatment_proxy[mask0].unsqueeze(-1),
            outcome_proxy=test_data_t.outcome_proxy[mask0].unsqueeze(-1),
            outcome=test_data_t.outcome[mask0].unsqueeze(-1),
            backdoor=test_data_t.backdoor[mask0].unsqueeze(-1),
            structural= test_data_t.structural
        )
        
        # 训练
        trainer1 = MMR_Trainer_Simulation(model_config1, random_seed)
        trainer0 = MMR_Trainer_Simulation(model_config0, random_seed)
        
        _, model1 = trainer1.train(train_data_1_t, val_data_1_t)
        _, model0 = trainer0.train(train_data_0_t, val_data_0_t)

        if trainer1.gpu_flg:
            # torch.cuda.empty_cache()
            test_data_t = test_data_t.to_gpu()
        
        '''
        test_data = MMRTrainDataSet_q(
            treatment=test_data.treatment,
            treatment_target=test_data.treatment_target,
            treatment_proxy=test_data.treatment_proxy,
            outcome_proxy=test_data.outcome_proxy,
            outcome=test_data.outcome,
            backdoor=test_data.backdoor
        )
        test_data_t = MMRTrainDataSetTorch_q.from_numpy(test_data)
        '''
        
        pred1 = trainer1.predict(model1, test_data_t_1).numpy() #q11(Z1,Y1,1) for 1
        pred0 = trainer0.predict(model0, test_data_t_0).numpy() #q11(Z1,Y1,-1) for -1
        
        '''
        #原输出
        res1 = test_data_t.outcome.squeeze(-1) * pred1.squeeze(-1) * (test_data_t.treatment.reshape(-1) == torch.ones(test_data_t.treatment.shape[0])).float()
        res0 = test_data_t.outcome.squeeze(-1) * pred0.squeeze(-1) * (test_data_t.treatment.reshape(-1) == -torch.ones(test_data_t.treatment.shape[0])).float() # 0 -> -1 
        res = [res1.mean().item() , res0.mean().item()]
        
        with open(file_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f'DeepMMR_{kind}_mod'] + res)
        '''
        return pred1, pred0

if __name__ == "__main__":    
    t1,t0 = run_mmr_experiments(num_runs=1, scenario='S1', kind='u', n_train=2000, n_test=1000, output_dir='./results/simulation')
    