import numpy as np
import pandas as pd
import json
import itertools
import torch
import pathlib
from pathlib import Path
from typing import Dict, Any
from skopt import gp_minimize
from skopt.space import Real, Integer

from src.data.data_class import MMRTrainDataSet_q, MMRTestDataSet_q, MMRTrainDataSetTorch_q, MMRTestDataSetTorch_q, MMRTrainDataSet_q_2stage, MMRTrainDataSetTorch_q_2stage

# from src.data.simulation import generate_train_simulation_q
from src.data.simu_new import generate_train_simulation_q, generate_train_simulation_q_2stage
from src.models.MMR.MMR_trainers import MMR_Trainer_Simulation, MMR_Trainer_Simulation_2stage
from src.utils.setseed import set_seed



def hp_search_simulation_1s(num_runs: int, scenario: str, kind: str, treatment1: int, n_train: int, n_test: int, output_dir: str):
        
    # Generate data
    train_data = generate_train_simulation_q(int(n_train * 0.75), scenario)
    val_data = generate_train_simulation_q(int(n_train * 0.25), scenario)
    
    if treatment1 == 1:
        train_data = MMRTrainDataSet_q(
            treatment=train_data.treatment,
            treatment_target=(train_data.treatment == 1), 
            treatment_proxy=train_data.treatment_proxy,
            outcome_proxy=train_data.outcome_proxy,
            outcome=train_data.outcome,
            backdoor=train_data.backdoor
        )
        train_data_t = MMRTrainDataSetTorch_q.from_numpy(train_data)
        
        val_data = MMRTrainDataSet_q(
            treatment=val_data.treatment,
            treatment_target=(val_data.treatment == 1),
            treatment_proxy=val_data.treatment_proxy,
            outcome_proxy=val_data.outcome_proxy,
            outcome=val_data.outcome,
            backdoor=val_data.backdoor
        )
        val_data_t = MMRTrainDataSetTorch_q.from_numpy(val_data)
        
    elif treatment1 == -1:
        train_data = MMRTrainDataSet_q(
            treatment=train_data.treatment,
            treatment_target=(train_data.treatment == -1), #0 -> -1
            treatment_proxy=train_data.treatment_proxy,
            outcome_proxy=train_data.outcome_proxy,
            outcome=train_data.outcome,
            backdoor=train_data.backdoor
        )
        train_data_t = MMRTrainDataSetTorch_q.from_numpy(train_data)
        
        val_data = MMRTrainDataSet_q(
            treatment=val_data.treatment,
            treatment_target=(val_data.treatment == -1), #0 -> -1
            treatment_proxy=val_data.treatment_proxy,
            outcome_proxy=val_data.outcome_proxy,
            outcome=val_data.outcome,
            backdoor=val_data.backdoor
        )
        val_data_t = MMRTrainDataSetTorch_q.from_numpy(val_data)
    
    
    # Define the hyperparameter space
    param_space = [
        Real(1e-7, 1e-2, prior='log-uniform', name='learning_rate'),
        Real(1e-7, 1e-2, prior='log-uniform', name='l2_penalty'),
        Real(0.1, 0.5, prior='uniform', name='dropout_prob'),
        Integer(3, 8, name='network_depth'),
        Integer(20, 60, name='network_width')
    ]
    keys = ['learning_rate', 'l2_penalty', 'dropout_prob', 'network_depth', 'network_width']

    def evaluate_model(params):
        model_config = {
            "n_epochs": 200,
            "batch_size": 100,
            "loss_name": f"{kind.upper()}_statistic"
        }
        model_config.update(dict(zip(keys, params)))
        
        trainer = MMR_Trainer_Simulation(model_config)
        loss, _ = trainer.train(train_data_t, val_data_t)
        
        return abs(loss)
    
    # Optimize
    best_params = gp_minimize(
        evaluate_model,
        param_space,
        n_calls=num_runs,
    )

    # optimal hyperparameters
    best_params_converted = {key: float(value) for key, value in zip(keys, best_params.x)}
    best_params_converted.update({
        "n_epochs": 200,
        "batch_size": 100,
        "loss_name": f"{kind.upper()}_statistic"
    })



    output_file_path = output_dir / f'deepmmr_a1={int(treatment1)}_1s.json'
    with open(output_file_path, 'w') as f:
        json.dump(best_params_converted, f, indent=4)
    
    print(f"hyper parameter search for a1={int(treatment1)} finished.")



def hp_search_simulation2(num_runs: int, scenario: str, kind: str, treatment1: int, treatment2: int, n_train: int, n_test: int, output_dir: str):
    
    # Generate data
    train_data = generate_train_simulation_q_2stage(int(n_train * 0.75), scenario)
    val_data = generate_train_simulation_q_2stage(int(n_train * 0.25), scenario)
    
    with open(Path(str(pathlib.Path(__file__).parent.parent.parent.parent / 'configs' / 'simulation' / f'deepmmr_a1={int(treatment1)}_1s.json')),'r') as f:
            model_config1 = json.load(f)
            
    if treatment1 == 1:
        if treatment2 == 1:
            train_data = MMRTrainDataSet_q_2stage(
                treatment1=train_data.treatment1,
                treatment_target1=(train_data.treatment1 == 1),
                treatment_proxy1=train_data.treatment_proxy1,
                outcome_proxy1=train_data.outcome_proxy1,
                outcome1=train_data.outcome1,
                backdoor1=train_data.backdoor1,
                treatment2=train_data.treatment2,
                treatment_target2=(train_data.treatment2 == 1),
                treatment_proxy2=train_data.treatment_proxy2,
                outcome2=train_data.outcome2,
                outcome_proxy2=train_data.outcome_proxy2,
                backdoor2=train_data.backdoor2
            )
            train_data_t = MMRTrainDataSetTorch_q_2stage.from_numpy(train_data)

            val_data = MMRTrainDataSet_q_2stage(
                treatment1=val_data.treatment1,
                treatment_target1=(val_data.treatment1 == 1),
                treatment_proxy1=val_data.treatment_proxy1,
                outcome_proxy1=val_data.outcome_proxy1,
                outcome1=val_data.outcome1,
                backdoor1=val_data.backdoor1,
                treatment2=val_data.treatment2,
                treatment_target2=(val_data.treatment2 == 1),
                treatment_proxy2=val_data.treatment_proxy2,
                outcome2=val_data.outcome2,
                outcome_proxy2=val_data.outcome_proxy2,
                backdoor2=val_data.backdoor2
            )
            val_data_t = MMRTrainDataSetTorch_q_2stage.from_numpy(val_data)
        
        elif treatment2 == -1:
            train_data = MMRTrainDataSet_q_2stage(
                treatment1=train_data.treatment1,
                treatment_target1=(train_data.treatment1 == 1),
                treatment_proxy1=train_data.treatment_proxy1,
                outcome_proxy1=train_data.outcome_proxy1,
                outcome1=train_data.outcome1,
                backdoor1=train_data.backdoor1,
                treatment2=train_data.treatment2,
                treatment_target2=(train_data.treatment2 == -1),
                treatment_proxy2=train_data.treatment_proxy2,
                outcome2=train_data.outcome2,
                outcome_proxy2=train_data.outcome_proxy2,
                backdoor2=train_data.backdoor2
            )
            train_data_t = MMRTrainDataSetTorch_q_2stage.from_numpy(train_data)

            val_data = MMRTrainDataSet_q_2stage(
                treatment1=val_data.treatment1,
                treatment_target1=(val_data.treatment1 == 1),
                treatment_proxy1=val_data.treatment_proxy1,
                outcome_proxy1=val_data.outcome_proxy1,
                outcome1=val_data.outcome1,
                backdoor1=val_data.backdoor1,
                treatment2=val_data.treatment2,
                treatment_target2=(val_data.treatment2 == -1),
                treatment_proxy2=val_data.treatment_proxy2,
                outcome2=val_data.outcome2,
                outcome_proxy2=val_data.outcome_proxy2,
                backdoor2=val_data.backdoor2
            )
            val_data_t = MMRTrainDataSetTorch_q_2stage.from_numpy(val_data)
    
    elif treatment1 == -1: 
        if treatment2 == 1:
            train_data = MMRTrainDataSet_q_2stage(
                treatment1=train_data.treatment1,
                treatment_target1=(train_data.treatment1 == -1),
                treatment_proxy1=train_data.treatment_proxy1,
                outcome_proxy1=train_data.outcome_proxy1,
                outcome1=train_data.outcome1,
                backdoor1=train_data.backdoor1,
                treatment2=train_data.treatment2,
                treatment_target2=(train_data.treatment2 == 1),
                treatment_proxy2=train_data.treatment_proxy2,
                outcome2=train_data.outcome2,
                outcome_proxy2=train_data.outcome_proxy2,
                backdoor2=train_data.backdoor2
            )
            train_data_t = MMRTrainDataSetTorch_q_2stage.from_numpy(train_data)

            val_data = MMRTrainDataSet_q_2stage(
                treatment1=val_data.treatment1,
                treatment_target1=(val_data.treatment1 == -1),
                treatment_proxy1=val_data.treatment_proxy1,
                outcome_proxy1=val_data.outcome_proxy1,
                outcome1=val_data.outcome1,
                backdoor1=val_data.backdoor1,
                treatment2=val_data.treatment2,
                treatment_target2=(val_data.treatment2 == 1),
                treatment_proxy2=val_data.treatment_proxy2,
                outcome2=val_data.outcome2,
                outcome_proxy2=val_data.outcome_proxy2,
                backdoor2=val_data.backdoor2
            )
            val_data_t = MMRTrainDataSetTorch_q_2stage.from_numpy(val_data)
        
        elif treatment2 == -1:
            train_data = MMRTrainDataSet_q_2stage(
                treatment1=train_data.treatment1,
                treatment_target1=(train_data.treatment1 == -1),
                treatment_proxy1=train_data.treatment_proxy1,
                outcome_proxy1=train_data.outcome_proxy1,
                outcome1=train_data.outcome1,
                backdoor1=train_data.backdoor1,
                treatment2=train_data.treatment2,
                treatment_target2=(train_data.treatment2 == -1),
                treatment_proxy2=train_data.treatment_proxy2,
                outcome2=train_data.outcome2,
                outcome_proxy2=train_data.outcome_proxy2,
                backdoor2=train_data.backdoor2
            )
            train_data_t = MMRTrainDataSetTorch_q_2stage.from_numpy(train_data)

            val_data = MMRTrainDataSet_q_2stage(
                treatment1=val_data.treatment1,
                treatment_target1=(val_data.treatment1 == -1),
                treatment_proxy1=val_data.treatment_proxy1,
                outcome_proxy1=val_data.outcome_proxy1,
                outcome1=val_data.outcome1,
                backdoor1=val_data.backdoor1,
                treatment2=val_data.treatment2,
                treatment_target2=(val_data.treatment2 == -1),
                treatment_proxy2=val_data.treatment_proxy2,
                outcome2=val_data.outcome2,
                outcome_proxy2=val_data.outcome_proxy2,
                backdoor2=val_data.backdoor2
            )
            val_data_t = MMRTrainDataSetTorch_q_2stage.from_numpy(val_data)
        
    param_space2 =[
        Real(1e-7, 1e-2, prior='log-uniform', name='learning_rate'),
        Real(1e-7, 1e-2, prior='log-uniform', name='l2_penalty'),
        Real(0.1, 0.5, prior='uniform', name='dropout_prob'),
        Integer(2, 8, name='network_depth'),
        Integer(20, 60, name='network_width')
    ]
    keys = ['learning_rate', 'l2_penalty', 'dropout_prob', 'network_depth', 'network_width']
    
    def eva_model(params):
        model_config2 = {
            "n_epochs": 200,
            "batch_size": 100,
            "loss_name": f"{kind.upper()}_statistic"
        }
        model_config2.update(dict(zip(keys, params)))
        
        trainer = MMR_Trainer_Simulation_2stage(model_config1 , model_config2)
        loss, _ = trainer.train2(train_data_t, val_data_t)
        
        return abs(loss)
    
    best_params = gp_minimize(
        eva_model,
        param_space2,
        n_calls=num_runs,
    )
    
    best_params_converted = {key: float(value) for key, value in zip(keys, best_params.x)}
    best_params_converted.update({
        "n_epochs": 100,
        "batch_size": 100,
        "loss_name": f"{kind.upper()}_statistic"
    })
    
    output_file_path = output_dir / f'deepmmr_a1={int(treatment1)}_a2={int(treatment2)}_2s.json'
    
    with open(output_file_path, 'w') as f:
        json.dump(best_params_converted, f, indent=4)
        
    print(f"hyper parameter search for a1={int(treatment1)}, a2={int(treatment2)} finished.")



def hp_search_simulation_2s(num_runs: int, scenario: str, kind: str, treatment1: int, treatment2: int, n_train: int, n_test: int, output_dir: str):
    # Generate data
    train_data = generate_train_simulation_q_2stage(int(n_train * 0.75), scenario)
    val_data = generate_train_simulation_q_2stage(int(n_train * 0.25), scenario)
    
    with open(Path(str(pathlib.Path(__file__).parent.parent.parent.parent / 'configs' / 'simulation' / f'deepmmr_a1={int(treatment1)}_1s.json')),'r') as f:
            model_config1 = json.load(f)
            
    if treatment1 == 1:
        if treatment2 == 1:
            train_data = MMRTrainDataSet_q_2stage(
                treatment1=train_data.treatment1,
                treatment_target1=(train_data.treatment1 == 1),
                treatment_proxy1=train_data.treatment_proxy1,
                outcome_proxy1=train_data.outcome_proxy1,
                outcome1=train_data.outcome1,
                backdoor1=train_data.backdoor1,
                treatment2=train_data.treatment2,
                treatment_target2=(train_data.treatment2 == 1),
                treatment_proxy2=train_data.treatment_proxy2,
                outcome2=train_data.outcome2,
                outcome_proxy2=train_data.outcome_proxy2,
                backdoor2=train_data.backdoor2
            )
            train_data_t = MMRTrainDataSetTorch_q_2stage.from_numpy(train_data)

            val_data = MMRTrainDataSet_q_2stage(
                treatment1=val_data.treatment1,
                treatment_target1=(val_data.treatment1 == 1),
                treatment_proxy1=val_data.treatment_proxy1,
                outcome_proxy1=val_data.outcome_proxy1,
                outcome1=val_data.outcome1,
                backdoor1=val_data.backdoor1,
                treatment2=val_data.treatment2,
                treatment_target2=(val_data.treatment2 == 1),
                treatment_proxy2=val_data.treatment_proxy2,
                outcome2=val_data.outcome2,
                outcome_proxy2=val_data.outcome_proxy2,
                backdoor2=val_data.backdoor2
            )
            val_data_t = MMRTrainDataSetTorch_q_2stage.from_numpy(val_data)
        
        elif treatment2 == -1:
            train_data = MMRTrainDataSet_q_2stage(
                treatment1=train_data.treatment1,
                treatment_target1=(train_data.treatment1 == 1),
                treatment_proxy1=train_data.treatment_proxy1,
                outcome_proxy1=train_data.outcome_proxy1,
                outcome1=train_data.outcome1,
                backdoor1=train_data.backdoor1,
                treatment2=train_data.treatment2,
                treatment_target2=(train_data.treatment2 == -1),
                treatment_proxy2=train_data.treatment_proxy2,
                outcome2=train_data.outcome2,
                outcome_proxy2=train_data.outcome_proxy2,
                backdoor2=train_data.backdoor2
            )
            train_data_t = MMRTrainDataSetTorch_q_2stage.from_numpy(train_data)

            val_data = MMRTrainDataSet_q_2stage(
                treatment1=val_data.treatment1,
                treatment_target1=(val_data.treatment1 == 1),
                treatment_proxy1=val_data.treatment_proxy1,
                outcome_proxy1=val_data.outcome_proxy1,
                outcome1=val_data.outcome1,
                backdoor1=val_data.backdoor1,
                treatment2=val_data.treatment2,
                treatment_target2=(val_data.treatment2 == -1),
                treatment_proxy2=val_data.treatment_proxy2,
                outcome2=val_data.outcome2,
                outcome_proxy2=val_data.outcome_proxy2,
                backdoor2=val_data.backdoor2
            )
            val_data_t = MMRTrainDataSetTorch_q_2stage.from_numpy(val_data)
    
    elif treatment1 == -1: 
        if treatment2 == 1:
            train_data = MMRTrainDataSet_q_2stage(
                treatment1=train_data.treatment1,
                treatment_target1=(train_data.treatment1 == -1),
                treatment_proxy1=train_data.treatment_proxy1,
                outcome_proxy1=train_data.outcome_proxy1,
                outcome1=train_data.outcome1,
                backdoor1=train_data.backdoor1,
                treatment2=train_data.treatment2,
                treatment_target2=(train_data.treatment2 == 1),
                treatment_proxy2=train_data.treatment_proxy2,
                outcome2=train_data.outcome2,
                outcome_proxy2=train_data.outcome_proxy2,
                backdoor2=train_data.backdoor2
            )
            train_data_t = MMRTrainDataSetTorch_q_2stage.from_numpy(train_data)

            val_data = MMRTrainDataSet_q_2stage(
                treatment1=val_data.treatment1,
                treatment_target1=(val_data.treatment1 == -1),
                treatment_proxy1=val_data.treatment_proxy1,
                outcome_proxy1=val_data.outcome_proxy1,
                outcome1=val_data.outcome1,
                backdoor1=val_data.backdoor1,
                treatment2=val_data.treatment2,
                treatment_target2=(val_data.treatment2 == 1),
                treatment_proxy2=val_data.treatment_proxy2,
                outcome2=val_data.outcome2,
                outcome_proxy2=val_data.outcome_proxy2,
                backdoor2=val_data.backdoor2
            )
            val_data_t = MMRTrainDataSetTorch_q_2stage.from_numpy(val_data)
        
        elif treatment2 == -1:
            train_data = MMRTrainDataSet_q_2stage(
                treatment1=train_data.treatment1,
                treatment_target1=(train_data.treatment1 == -1),
                treatment_proxy1=train_data.treatment_proxy1,
                outcome_proxy1=train_data.outcome_proxy1,
                outcome1=train_data.outcome1,
                backdoor1=train_data.backdoor1,
                treatment2=train_data.treatment2,
                treatment_target2=(train_data.treatment2 == -1),
                treatment_proxy2=train_data.treatment_proxy2,
                outcome2=train_data.outcome2,
                outcome_proxy2=train_data.outcome_proxy2,
                backdoor2=train_data.backdoor2
            )
            train_data_t = MMRTrainDataSetTorch_q_2stage.from_numpy(train_data)

            val_data = MMRTrainDataSet_q_2stage(
                treatment1=val_data.treatment1,
                treatment_target1=(val_data.treatment1 == -1),
                treatment_proxy1=val_data.treatment_proxy1,
                outcome_proxy1=val_data.outcome_proxy1,
                outcome1=val_data.outcome1,
                backdoor1=val_data.backdoor1,
                treatment2=val_data.treatment2,
                treatment_target2=(val_data.treatment2 == -1),
                treatment_proxy2=val_data.treatment_proxy2,
                outcome2=val_data.outcome2,
                outcome_proxy2=val_data.outcome_proxy2,
                backdoor2=val_data.backdoor2
            )
            val_data_t = MMRTrainDataSetTorch_q_2stage.from_numpy(val_data)
        
    param_space2 =[
        Real(1e-7, 1e-2, prior='log-uniform', name='learning_rate'),
        Real(1e-7, 1e-2, prior='log-uniform', name='l2_penalty'),
        Real(0.1, 0.5, prior='uniform', name='dropout_prob'),
        Integer(2, 8, name='network_depth'),
        Integer(20, 60, name='network_width')
    ]
    keys = ['learning_rate', 'l2_penalty', 'dropout_prob', 'network_depth', 'network_width']
    
    def eva_model_1(params):
        model_config2 = {
            "n_epochs": 200,
            "batch_size": 100,
            "loss_name": f"{kind.upper()}_statistic"
        }
        model_config2.update(dict(zip(keys, params)))
        
        trainer = MMR_Trainer_Simulation_2stage(model_config1, model_config2)
        loss, _ = trainer.train_2s(train_data_t, val_data_t)
        
        return abs(loss[0]) #a1=1
    
    def eva_model_0(params):
        model_config2 = {
            "n_epochs": 200,
            "batch_size": 100,
            "loss_name": f"{kind.upper()}_statistic"
        }
        model_config2.update(dict(zip(keys, params)))
        
        trainer = MMR_Trainer_Simulation_2stage(model_config1, model_config2)
        loss, _ = trainer.train_2s(train_data_t, val_data_t)
        
        return abs(loss[1]) #a1=-1
    
    best_params_1 = gp_minimize(
        eva_model_1,
        param_space2,
        n_calls=num_runs,
    )
    best_params_0 = gp_minimize(
        eva_model_0,
        param_space2,
        n_calls=num_runs,
    )
    # tune 1
    best_params_converted_1 = {key: float(value) for key, value in zip(keys, best_params_1.x)}
    best_params_converted_1.update({
        "n_epochs": 200,
        "batch_size": 100,
        "loss_name": f"{kind.upper()}_statistic"
    })
    
    output_file_path1 = output_dir / f'deepmmr_a1=1_a2={int(treatment2)}_2s.json'
    
    with open(output_file_path1, 'w') as f:
        json.dump(best_params_converted_1, f, indent=4)
        
    print(f"hyper parameter search for a1=1, a2={int(treatment2)} finished.")
    
    # tune 0
    best_params_converted_0 = {key: float(value) for key, value in zip(keys, best_params_0.x)}
    best_params_converted_0.update({
        "n_epochs": 200,
        "batch_size": 100,
        "loss_name": f"{kind.upper()}_statistic"
    })
    
    output_file_path0 = output_dir / f'deepmmr_a1=-1_a2={int(treatment2)}_2s.json'
    
    with open(output_file_path0, 'w') as f:
        json.dump(best_params_converted_0, f, indent=4)
        
    print(f"hyper parameter search for a1=-1, a2={int(treatment2)} finished.")




def main_hps(num_runs: int, n_trains: int, n_tests: int, output_dir: str = "./configs/simulation"):
    set_seed(2048)
    
    # 需要搜索的参数
    n_experiments  = num_runs         # gp_minimize 调用次数
    n_train    = n_trains             # 训练集大小
    n_test     = n_tests            # 测试集大小
    output_dir = Path(output_dir)   # 存放 json 的目录
    
    treatment_set_1s = {-1,1}
    treatment_set_2s = {(1,1),(1,-1),(-1,1),(-1,-1)}  #1 表示治疗组，-1 表示对照组
    
    
    for treatment1 in treatment_set_1s:
        hp_search_simulation_1s(
            num_runs=n_experiments,
            scenario="S1",
            kind="u",
            treatment1=treatment1,
            n_train=n_train,
            n_test=n_tests,
            output_dir=output_dir
        ) 
    
    for (treatment1,treatment2) in treatment_set_2s: 
    
        hp_search_simulation_2s(
            num_runs=n_experiments, 
            scenario="S1", 
            kind="u", 
            treatment1=treatment1, 
            treatment2=treatment2, 
            n_train=n_train, 
            n_test=n_test, 
            output_dir=output_dir)



if __name__ == "__main__":
    n_experiments = 1
    n_train = 2000
    n_test = 1000
    
    main_hps(num_runs=n_experiments, n_trains=n_train, n_tests=n_test)