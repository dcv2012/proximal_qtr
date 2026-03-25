import os.path as op
from typing import Optional, Dict, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm, gui
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from src.data.data_class import MMRTrainDataSet_q, MMRTestDataSet_q
from src.data.data_class import MMRTrainDataSetTorch_q, MMRTestDataSetTorch_q, MMRTrainDataSetTorch_q_2stage, MMRTestDataSetTorch_q_2stage

from src.models.MMR.MMR_loss import MMR_loss
from src.models.MMR.MMR_model import MLP_for_MMR
from src.models.MMR.kernel_utils import calculate_kernel_matrix



class EarlyStopping:
    def __init__(self, patience=20, delta=1e-4):
        """
        初始化早停机制。
        
        参数:
        - patience (int): 允许验证损失不改善的最大轮数。
        - delta (float): 改善阈值，小于此值视为无改善。
        """
        
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        调用早停检查。
        
        参数:
        - val_loss (float): 当前验证损失。
        
        返回:
        - 无。
        
        功能:
        - 更新最佳分数和计数器。
        - 如果无改善达到耐心值，设置 early_stop 为 True。
        """
        
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class MMR_Trainer_Simulation:
    def __init__(self, train_params: Dict[str, Any]):
        """
        初始化模拟数据训练器。
        
        参数:
        - train_params (Dict[str, Any]): 训练参数字典，包括 n_epochs、batch_size 等。
        - random_seed (int): 随机种子。
        """
        
        self.train_params = train_params
        self.n_epochs = train_params['n_epochs']
        self.batch_size = train_params['batch_size']
        self.l2_penalty = train_params['l2_penalty']
        self.learning_rate = train_params['learning_rate']
        self.loss_name = train_params['loss_name']
        self.gpu_flg = torch.cuda.is_available()

    def compute_kernel(self, kernel_inputs):
        """
        计算核矩阵。
        
        参数:
        - kernel_inputs (torch.Tensor): 核输入数据。
        
        返回:
        - torch.Tensor: 高斯核矩阵。
        """
        
        return calculate_kernel_matrix(kernel_inputs)
    
    def train(self, train_t: MMRTrainDataSetTorch_q, val_t: MMRTrainDataSetTorch_q):
        """
        训练 MMR 模型。
        
        参数:
        - train_t (MMRTrainDataSetTorch_q): 训练集。
        - val_t (MMRTrainDataSetTorch_q): 验证集。
        
        返回:
        - tuple: (最佳验证损失, 训练好的模型)。
        
        功能:
        - 初始化模型、优化器和调度器。
        - 执行批次训练，使用 MMR_loss 计算损失。
        - 应用早停和学习率调度。
        """
        
        input_size = 1 + train_t.backdoor.shape[1]
        model = MLP_for_MMR(input_dim=input_size, train_params=self.train_params)
    
        if self.gpu_flg:
            train_t = train_t.to_gpu()
            val_t = val_t.to_gpu()
            model.cuda()
    
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.l2_penalty)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
                                                   patience=10, threshold=1e-4)
   

        train_a, train_w, train_x, train_y, train_z, train_tt = train_t.treatment, train_t.outcome_proxy, train_t.backdoor, train_t.outcome, train_t.treatment_proxy, train_t.treatment_target
        val_a, val_w, val_x, val_y, val_z, val_tt = val_t.treatment, val_t.outcome_proxy, val_t.backdoor, val_t.outcome, val_t.treatment_proxy, val_t.treatment_target
    
        # batch training
        train_dataset = TensorDataset(train_a, train_w, train_x, train_y, train_z, train_tt)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
    
        early_stopping = EarlyStopping(patience=20, delta=1e-4)
        best_model_state = None
        best_val_loss = float('inf')
    
        for epoch in tqdm(range(self.n_epochs), desc="Stage 1 Training", leave=True):
            # train
            model.train()
            total_loss = 0
    
            for batch_a, batch_w, batch_x, batch_y, batch_z, batch_tt in train_loader:
                if self.gpu_flg:
                    batch_a, batch_w, batch_x, batch_y, batch_z, batch_tt = batch_a.cuda(), batch_w.cuda(), batch_x.cuda(), batch_y.cuda(), batch_z.cuda(), batch_tt.cuda()

                optimizer.zero_grad()
                batch_inputs = torch.cat((batch_z, batch_x), dim=1)
                pred = model(batch_inputs)
    
                kernel_inputs = torch.cat((batch_w, batch_x), dim=1)
                kernel_matrix = self.compute_kernel(kernel_inputs)
    
                loss = torch.abs(MMR_loss(pred * batch_tt, torch.ones_like(batch_y), kernel_matrix, self.loss_name))
                
                loss.backward()
                optimizer.step()
    
                total_loss += loss.item()
    
    
            # eval
            model.eval()
            with torch.no_grad():
                val_inputs = torch.cat((val_z, val_x), dim=1)
                val_pred = model(val_inputs)
    
                val_kernel_inputs = torch.cat((val_w, val_x), dim=1)
                val_kernel_matrix = self.compute_kernel(val_kernel_inputs)
    
                val_loss = abs(MMR_loss(val_pred * val_tt, torch.ones_like(val_y), val_kernel_matrix, self.loss_name).detach().cpu().item())
    
            scheduler.step(val_loss)
    
            # early stop
            early_stopping(val_loss)
            if early_stopping.early_stop:
                break
    
            # update the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
    
        # load the best model
        model.load_state_dict(best_model_state)
    
        return best_val_loss, model


    @staticmethod
    def predict(model, test_data_t):
        """
        使用训练好的模型进行预测。
        
        参数:
        - model (nn.Module): 训练好的 MMR 模型。
        - test_data_t: Tensor测试集。
        
        返回:
        - torch.Tensor: 预测结果。
        
        功能:
        - 拼接输入数据，运行模型。
        """
        
        device = next(model.parameters()).device  # 获取模型设备
        tempZ = test_data_t.treatment_proxy.to(device) #to
        tempX = test_data_t.backdoor.to(device) #to
        model_inputs_test = torch.cat((tempZ, tempX),dim = 1)

        with torch.no_grad():
            E_zx = model(model_inputs_test)

        return E_zx.cpu()
    
    

class MMR_Trainer_Simulation_2stage(MMR_Trainer_Simulation):
    def __init__(self, train_params1: Dict[str, Any], train_params2: Dict[str, Any]):
        super().__init__(train_params1)

        self.train_params2 = train_params2
        self.n_epochs2 = train_params2['n_epochs']
        self.batch_size2 = train_params2['batch_size']
        self.l2_penalty2 = train_params2['l2_penalty']
        self.learning_rate2 = train_params2['learning_rate']
        self.loss_name2 = train_params2['loss_name']
        
    def train2(self, train_t: MMRTrainDataSetTorch_q_2stage, val_t: MMRTrainDataSetTorch_q_2stage):
        # 这里可以实现两阶段训练的逻辑，或者根据需要覆盖父类的 train 方法
        # 例如，第一阶段训练 A1 的模型，第二阶段训练 A2 的模型
        
        ## input_size = 1 + train_t.backdoor2.shape[1]
        input_size = train_t.treatment_proxy1.shape[1]+ train_t.treatment_proxy2.shape[1] + train_t.backdoor1.shape[1] + train_t.backdoor2.shape[1]
        model = MLP_for_MMR(input_dim=input_size, train_params=self.train_params2)
        
        if self.gpu_flg:
            train_t = train_t.to_gpu()
            val_t = val_t.to_gpu()
            model.cuda()
            
        optimizer= optim.AdamW(model.parameters(), lr=self.learning_rate2, weight_decay=self.l2_penalty2)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
                                                   patience=10, threshold=1e-4)
    
        train_a1, train_w1, train_y0, train_y1, train_z1, train_tt1 = train_t.treatment1, train_t.outcome_proxy1, train_t.backdoor1, train_t.outcome1, train_t.treatment_proxy1, train_t.treatment_target1
        train_a2, train_w2, train_y2, train_z2, train_tt2 = train_t.treatment2, train_t.outcome_proxy2, train_t.outcome2, train_t.treatment_proxy2, train_t.treatment_target2
        val_a1, val_w1, val_y0, val_y1, val_z1, val_tt1 = val_t.treatment1, val_t.outcome_proxy1, val_t.backdoor1, val_t.outcome1, val_t.treatment_proxy1, val_t.treatment_target1
        val_a2, val_w2, val_y2, val_z2, val_tt2 = val_t.treatment2, val_t.outcome_proxy2, val_t.outcome2, val_t.treatment_proxy2, val_t.treatment_target2
        
        train_dataset = TensorDataset(train_a1, train_w1, train_y0, train_y1, train_z1, train_tt1, train_a2, train_w2, train_y2, train_z2, train_tt2)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size2, shuffle=True)
        
        early_stopping = EarlyStopping(patience=20, delta=1e-4)
        best_model_state = None
        best_val_loss = float('inf')
        device = next(model.parameters()).device
        
        # 训练第一阶段的q11
        train_data_1t = MMRTrainDataSetTorch_q(
            treatment=train_a1.requires_grad_(True),
            treatment_target=(train_a1 == 1).float().requires_grad_(True),
            treatment_proxy=train_z1.requires_grad_(True),
            outcome_proxy=train_w1.requires_grad_(True),
            outcome=train_y1.requires_grad_(True),
            backdoor=train_y0.requires_grad_(True)                   
        )
        train_data_0t = MMRTrainDataSetTorch_q(
            treatment=train_a1.requires_grad_(True),
            treatment_target=(train_a1 == -1).float().requires_grad_(True),
            treatment_proxy=train_z1.requires_grad_(True),
            outcome_proxy=train_w1.requires_grad_(True),
            outcome=train_y1.requires_grad_(True),
            backdoor=train_y0.requires_grad_(True)
        )
        val_data_1t = MMRTrainDataSetTorch_q(
            treatment=val_a1.requires_grad_(True),
            treatment_target=(val_a1 == 1).float().requires_grad_(True),
            treatment_proxy=val_z1.requires_grad_(True),
            outcome_proxy=val_w1.requires_grad_(True),
            outcome=val_y1.requires_grad_(True),
            backdoor=val_y0.requires_grad_(True)
        )
        val_data_0t = MMRTrainDataSetTorch_q(
            treatment=val_a1.requires_grad_(True),
            treatment_target=(val_a1 == -1).float().requires_grad_(True),
            treatment_proxy=val_z1.requires_grad_(True),
            outcome_proxy=val_w1.requires_grad_(True),
            outcome=val_y1.requires_grad_(True),
            backdoor=val_y0.requires_grad_(True)
        )
        
        q11_trainer1 = MMR_Trainer_Simulation(train_params=self.train_params) 
        q11_trainer0 = MMR_Trainer_Simulation(train_params=self.train_params)
        _, q11_model1 = q11_trainer1.train(train_data_1t, val_data_1t) #训练好的q11_1模型
        _, q11_model0 = q11_trainer0.train(train_data_0t, val_data_0t) #训练好的q11_0模型
        
        
        for epoch in tqdm(range(self.n_epochs2), desc="Stage 2 Training"):
            # train stage 1
            model.train()           
            total_loss = 0
    
            for batch_a1, batch_w1, batch_y0, batch_y1, batch_z1, batch_tt1, batch_a2, batch_w2, batch_y2, batch_z2, batch_tt2 in train_loader:
                if self.gpu_flg:
                    batch_a1, batch_w1, batch_y0, batch_y1, batch_z1, batch_tt1, batch_a2, batch_w2, batch_y2, batch_z2, batch_tt2 = batch_a1.cuda(), batch_w1.cuda(), batch_y0.cuda(), batch_y1.cuda(), batch_z1.cuda(), batch_tt1.cuda(), batch_a2.cuda(), batch_w2.cuda(), batch_y2.cuda(), batch_z2.cuda(), batch_tt2.cuda()

                batch_mask1 = (batch_a1 == 1)
                batch_mask0 = (batch_a1 == -1)
                
                # _x_xx_1 表示a1=1的子集，_x_xx_0 表示a1=-1的子集
                (batch_w1_1,batch_y0_1,batch_y1_1, batch_z1_1,batch_tt1_1,batch_a2_1,batch_w2_1,batch_y2_1,batch_z2_1,batch_tt2_1) = ([t[batch_mask1].unsqueeze(-1) for t in (batch_w1,batch_y0,batch_y1,batch_z1,batch_tt1,batch_a2,batch_w2,batch_y2,batch_z2,batch_tt2)])
                (batch_w1_0,batch_y0_0,batch_y1_0, batch_z1_0,batch_tt1_0,batch_a2_0,batch_w2_0,batch_y2_0,batch_z2_0,batch_tt2_0) = ([t[batch_mask0].unsqueeze(-1) for t in (batch_w1,batch_y0,batch_y1,batch_z1,batch_tt1,batch_a2,batch_w2,batch_y2,batch_z2,batch_tt2)])
                
                optimizer.zero_grad()
                
                # a1=1子集
                batch_inputs2_1 = torch.cat((batch_z1_1, batch_z2_1, batch_y0_1, batch_y1_1), dim=1)
                pred2_1 = model(batch_inputs2_1)
                kernel_inputs2_1 = torch.cat((batch_w1_1, batch_w2_1, batch_y0_1, batch_y1_1), dim=1)    # is a1 needed?
                kernel_matrix2_1 = self.compute_kernel(kernel_inputs2_1)
                q11_inputs_1 = torch.cat((batch_z1_1, batch_y0_1), dim=1).to(device) ##
                with torch.no_grad():
                    q11_pred_1 = q11_model1(q11_inputs_1)
                    
                loss2_1 = torch.abs(MMR_loss(pred2_1 * batch_tt2_1, q11_pred_1, kernel_matrix2_1, self.loss_name2))
                
                
                # a1=-1子集
                batch_inputs2_0 = torch.cat((batch_z1_0, batch_z2_0, batch_y0_0, batch_y1_0), dim=1)
                pred2_0 = model(batch_inputs2_0)
                kernel_inputs2_0 = torch.cat((batch_w1_0, batch_w2_0, batch_y0_0, batch_y1_0), dim=1)    # is a1 needed?
                kernel_matrix2_0 = self.compute_kernel(kernel_inputs2_0)
                q11_inputs_0 = torch.cat((batch_z1_0, batch_y0_0), dim=1).to(device) ##
                with torch.no_grad():
                    q11_pred_0 = q11_model0(q11_inputs_0)
                    
                loss2_0 = torch.abs(MMR_loss(pred2_0 * batch_tt2_0, q11_pred_0, kernel_matrix2_0, self.loss_name2))


                # 总损失是两个子集损失的平均
                loss = loss2_1 + loss2_0
                
                loss.backward()
                optimizer.step()
    
                total_loss += loss.item()
    
    
            # eval
            model.eval()
            
            val_mask1 = (val_a1 == 1)
            val_mask0 = (val_a1 == -1)
            (val_w1_1,val_y0_1,val_y1_1, val_z1_1,val_tt1_1,val_a2_1,val_w2_1,val_y2_1,val_z2_1,val_tt2_1) = ([t[val_mask1].unsqueeze(-1) for t in (val_w1,val_y0,val_y1,val_z1,val_tt1,val_a2,val_w2,val_y2,val_z2,val_tt2)])
            (val_w1_0,val_y0_0,val_y1_0, val_z1_0,val_tt1_0,val_a2_0,val_w2_0,val_y2_0,val_z2_0,val_tt2_0) = ([t[val_mask0].unsqueeze(-1) for t in (val_w1,val_y0,val_y1,val_z1,val_tt1,val_a2,val_w2,val_y2,val_z2,val_tt2)])
            
            val_inputs2_1 = torch.cat((val_z1_1, val_z2_1, val_y0_1, val_y1_1), dim=1)
            with torch.no_grad():
                val_pred2_1 = model(val_inputs2_1)
            val_kernel_inputs2_1 = torch.cat((val_w1_1, val_w2_1, val_y0_1, val_y1_1), dim=1)
            val_kernel_matrix2_1 = self.compute_kernel(val_kernel_inputs2_1)
            val_q11_inputs_1 = torch.cat((val_z1_1, val_y0_1), dim=1).to(device)
            with torch.no_grad():
                val_q11_pred_1 = q11_model1(val_q11_inputs_1)
            
            val_loss2_1 = abs(MMR_loss(val_pred2_1 * val_tt2_1, val_q11_pred_1, val_kernel_matrix2_1, self.loss_name2).detach().cpu().item())
        
            val_inputs2_0 = torch.cat((val_z1_0, val_z2_0, val_y0_0, val_y1_0), dim=1)
            with torch.no_grad():
                val_pred2_0 = model(val_inputs2_0)
            val_kernel_inputs2_0 = torch.cat((val_w1_0, val_w2_0, val_y0_0, val_y1_0), dim=1)
            val_kernel_matrix2_0 = self.compute_kernel(val_kernel_inputs2_0)
            val_q11_inputs_0 = torch.cat((val_z1_0, val_y0_0), dim=1).to(device)
            with torch.no_grad():
                val_q11_pred_0 = q11_model0(val_q11_inputs_0)
            
            val_loss2_0 = abs(MMR_loss(val_pred2_0 * val_tt2_0, val_q11_pred_0, val_kernel_matrix2_0, self.loss_name2).detach().cpu().item())
            
            # sum up
            val_loss = val_loss2_1 + val_loss2_0
            
            
            scheduler.step(val_loss)
    
            # early stop
            early_stopping(val_loss)
            if early_stopping.early_stop:
                break
    
            # update the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                
        model.load_state_dict(best_model_state)
            
        return best_val_loss, model
    
    
    
    def train_2s(self, train_t: MMRTrainDataSetTorch_q_2stage, val_t: MMRTrainDataSetTorch_q_2stage):
        input_size = train_t.treatment_proxy1.shape[1]+ train_t.treatment_proxy2.shape[1] + train_t.backdoor1.shape[1] + train_t.backdoor2.shape[1]
        model1 = MLP_for_MMR(input_dim=input_size, train_params=self.train_params2)
        model0 = MLP_for_MMR(input_dim=input_size, train_params=self.train_params2)
        
        if self.gpu_flg:
            train_t = train_t.to_gpu()
            val_t = val_t.to_gpu()
            model1.cuda()
            model0.cuda()
            
        optimizer1= optim.AdamW(model1.parameters(), lr=self.learning_rate2, weight_decay=self.l2_penalty2)
        optimizer0= optim.AdamW(model0.parameters(), lr=self.learning_rate2, weight_decay=self.l2_penalty2)
        
        scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min', factor=0.2,
                                                   patience=10, threshold=1e-4)
        scheduler0 = optim.lr_scheduler.ReduceLROnPlateau(optimizer0, mode='min', factor=0.2,
                                                   patience=10, threshold=1e-4)
    
        train_a1, train_w1, train_y0, train_y1, train_z1, train_tt1 = train_t.treatment1, train_t.outcome_proxy1, train_t.backdoor1, train_t.outcome1, train_t.treatment_proxy1, train_t.treatment_target1
        train_a2, train_w2, train_y2, train_z2, train_tt2 = train_t.treatment2, train_t.outcome_proxy2, train_t.outcome2, train_t.treatment_proxy2, train_t.treatment_target2
        val_a1, val_w1, val_y0, val_y1, val_z1, val_tt1 = val_t.treatment1, val_t.outcome_proxy1, val_t.backdoor1, val_t.outcome1, val_t.treatment_proxy1, val_t.treatment_target1
        val_a2, val_w2, val_y2, val_z2, val_tt2 = val_t.treatment2, val_t.outcome_proxy2, val_t.outcome2, val_t.treatment_proxy2, val_t.treatment_target2
        
        train_dataset = TensorDataset(train_a1, train_w1, train_y0, train_y1, train_z1, train_tt1, train_a2, train_w2, train_y2, train_z2, train_tt2)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size2, shuffle=True)
        
        early_stopping1 = EarlyStopping(patience=20, delta=1e-4)
        early_stopping0 = EarlyStopping(patience=20, delta=1e-4)
        
        best_model_state1 = None
        best_model_state0 = None
                
        best_val_loss1 = float('inf')
        best_val_loss0 = float('inf')
        
        # 训练第一阶段的q11
        train_data_1t = MMRTrainDataSetTorch_q(
            treatment=train_a1.requires_grad_(True),
            treatment_target=(train_a1 == 1).float().requires_grad_(True),
            treatment_proxy=train_z1.requires_grad_(True),
            outcome_proxy=train_w1.requires_grad_(True),
            outcome=train_y1.requires_grad_(True),
            backdoor=train_y0.requires_grad_(True)                   
        )
        train_data_0t = MMRTrainDataSetTorch_q(
            treatment=train_a1.requires_grad_(True),
            treatment_target=(train_a1 == -1).float().requires_grad_(True),
            treatment_proxy=train_z1.requires_grad_(True),
            outcome_proxy=train_w1.requires_grad_(True),
            outcome=train_y1.requires_grad_(True),
            backdoor=train_y0.requires_grad_(True)
        )
        val_data_1t = MMRTrainDataSetTorch_q(
            treatment=val_a1.requires_grad_(True),
            treatment_target=(val_a1 == 1).float().requires_grad_(True),
            treatment_proxy=val_z1.requires_grad_(True),
            outcome_proxy=val_w1.requires_grad_(True),
            outcome=val_y1.requires_grad_(True),
            backdoor=val_y0.requires_grad_(True)
        )
        val_data_0t = MMRTrainDataSetTorch_q(
            treatment=val_a1.requires_grad_(True),
            treatment_target=(val_a1 == -1).float().requires_grad_(True),
            treatment_proxy=val_z1.requires_grad_(True),
            outcome_proxy=val_w1.requires_grad_(True),
            outcome=val_y1.requires_grad_(True),
            backdoor=val_y0.requires_grad_(True)
        )
        
        q11_trainer1 = MMR_Trainer_Simulation(train_params=self.train_params) 
        q11_trainer0 = MMR_Trainer_Simulation(train_params=self.train_params)
        _, q11_model1 = q11_trainer1.train(train_data_1t, val_data_1t) #训练好的q11_1模型
        _, q11_model0 = q11_trainer0.train(train_data_0t, val_data_0t) #训练好的q11_0模型
        
        
        for epoch in tqdm(range(self.n_epochs2), desc="Stage 2 Training"):
            # train stage 1
            model1.train()
            model0.train()            
            total_loss1 = 0
            total_loss0 = 0
    
            for batch_a1, batch_w1, batch_y0, batch_y1, batch_z1, batch_tt1, batch_a2, batch_w2, batch_y2, batch_z2, batch_tt2 in train_loader:
                if self.gpu_flg:
                    batch_a1, batch_w1, batch_y0, batch_y1, batch_z1, batch_tt1, batch_a2, batch_w2, batch_y2, batch_z2, batch_tt2 = batch_a1.cuda(), batch_w1.cuda(), batch_y0.cuda(), batch_y1.cuda(), batch_z1.cuda(), batch_tt1.cuda(), batch_a2.cuda(), batch_w2.cuda(), batch_y2.cuda(), batch_z2.cuda(), batch_tt2.cuda()

                batch_mask1 = (batch_a1 == 1)
                batch_mask0 = (batch_a1 == -1)
                
                # _x_xx_1 表示a1=1的子集，_x_xx_0 表示a1=-1的子集
                (batch_w1_1,batch_y0_1,batch_y1_1, batch_z1_1,batch_tt1_1,batch_a2_1,batch_w2_1,batch_y2_1,batch_z2_1,batch_tt2_1) = ([t[batch_mask1].unsqueeze(-1) for t in (batch_w1,batch_y0,batch_y1,batch_z1,batch_tt1,batch_a2,batch_w2,batch_y2,batch_z2,batch_tt2)])
                (batch_w1_0,batch_y0_0,batch_y1_0, batch_z1_0,batch_tt1_0,batch_a2_0,batch_w2_0,batch_y2_0,batch_z2_0,batch_tt2_0) = ([t[batch_mask0].unsqueeze(-1) for t in (batch_w1,batch_y0,batch_y1,batch_z1,batch_tt1,batch_a2,batch_w2,batch_y2,batch_z2,batch_tt2)])
                
                optimizer1.zero_grad()
                optimizer0.zero_grad()
                
                # a1=1子集
                batch_inputs2_1 = torch.cat((batch_z1_1, batch_z2_1, batch_y0_1, batch_y1_1), dim=1)
                pred2_1 = model1(batch_inputs2_1)
                kernel_inputs2_1 = torch.cat((batch_w1_1, batch_w2_1, batch_y0_1, batch_y1_1), dim=1)    # is a1 needed?
                kernel_matrix2_1 = self.compute_kernel(kernel_inputs2_1)
                q11_inputs_1 = torch.cat((batch_z1_1, batch_y0_1), dim=1) ##
                with torch.no_grad():
                    q11_pred_1 = q11_model1(q11_inputs_1)
                    
                loss2_1 = torch.abs(MMR_loss(pred2_1 * batch_tt2_1, q11_pred_1, kernel_matrix2_1, self.loss_name2))
                loss2_1.backward()
                optimizer1.step()
                total_loss1 += loss2_1.item()
                
                # a1=-1子集
                batch_inputs2_0 = torch.cat((batch_z1_0, batch_z2_0, batch_y0_0, batch_y1_0), dim=1)
                pred2_0 = model0(batch_inputs2_0)
                kernel_inputs2_0 = torch.cat((batch_w1_0, batch_w2_0, batch_y0_0, batch_y1_0), dim=1)    # is a1 needed?
                kernel_matrix2_0 = self.compute_kernel(kernel_inputs2_0)
                q11_inputs_0 = torch.cat((batch_z1_0, batch_y0_0), dim=1)
                with torch.no_grad():
                    q11_pred_0 = q11_model0(q11_inputs_0)
                    
                loss2_0 = torch.abs(MMR_loss(pred2_0 * batch_tt2_0, q11_pred_0, kernel_matrix2_0, self.loss_name2))
                loss2_0.backward()
                optimizer0.step()
                total_loss0 += loss2_0.item()
    
    
            # eval
            model1.eval()
            model0.eval()
            
            val_mask1 = (val_a1 == 1)
            val_mask0 = (val_a1 == -1)
            (val_w1_1,val_y0_1,val_y1_1, val_z1_1,val_tt1_1,val_a2_1,val_w2_1,val_y2_1,val_z2_1,val_tt2_1) = ([t[val_mask1].unsqueeze(-1) for t in (val_w1,val_y0,val_y1,val_z1,val_tt1,val_a2,val_w2,val_y2,val_z2,val_tt2)])
            (val_w1_0,val_y0_0,val_y1_0, val_z1_0,val_tt1_0,val_a2_0,val_w2_0,val_y2_0,val_z2_0,val_tt2_0) = ([t[val_mask0].unsqueeze(-1) for t in (val_w1,val_y0,val_y1,val_z1,val_tt1,val_a2,val_w2,val_y2,val_z2,val_tt2)])
            
            with torch.no_grad():
                val_inputs2_1 = torch.cat((val_z1_1, val_z2_1, val_y0_1, val_y1_1), dim=1)
                val_pred2_1 = model1(val_inputs2_1)
                val_kernel_inputs2_1 = torch.cat((val_w1_1, val_w2_1, val_y0_1, val_y1_1), dim=1)
                val_kernel_matrix2_1 = self.compute_kernel(val_kernel_inputs2_1)
                val_q11_inputs_1 = torch.cat((val_z1_1, val_y0_1), dim=1)
                val_q11_pred_1 = q11_model1(val_q11_inputs_1)    
                val_loss2_1 = abs(MMR_loss(val_pred2_1 * val_tt2_1, val_q11_pred_1, val_kernel_matrix2_1, self.loss_name2).detach().cpu().item())
            
                val_inputs2_0 = torch.cat((val_z1_0, val_z2_0, val_y0_0, val_y1_0), dim=1)
                val_pred2_0 = model0(val_inputs2_0)
                val_kernel_inputs2_0 = torch.cat((val_w1_0, val_w2_0, val_y0_0, val_y1_0), dim=1)
                val_kernel_matrix2_0 = self.compute_kernel(val_kernel_inputs2_0)
                val_q11_inputs_0 = torch.cat((val_z1_0, val_y0_0), dim=1)
                val_q11_pred_0 = q11_model0(val_q11_inputs_0)
                val_loss2_0 = abs(MMR_loss(val_pred2_0 * val_tt2_0, val_q11_pred_0, val_kernel_matrix2_0, self.loss_name2).detach().cpu().item())
            
            
            scheduler1.step(val_loss2_1)
            scheduler0.step(val_loss2_0)
    
            # early stop
            early_stopping1(val_loss2_1)
            early_stopping0(val_loss2_0)
            if early_stopping1.early_stop and early_stopping0.early_stop:
                break
    
            # update the best model
            if val_loss2_1 < best_val_loss1:
                best_val_loss1 = val_loss2_1
                best_model_state1 = model1.state_dict()
            if val_loss2_0 < best_val_loss0:
                best_val_loss0 = val_loss2_0
                best_model_state0 = model0.state_dict()
                
        model1.load_state_dict(best_model_state1)   # a1=1
        model0.load_state_dict(best_model_state0)   # a1=-1
            
        return [best_val_loss1, best_val_loss0], [model1, model0]
        
        
    @staticmethod
    def predict_2s(model, test_data_t) -> torch.Tensor:
        device = next(model.parameters()).device  
        
        tempZ1 = test_data_t.treatment_proxy1.to(device)
        tempZ2 = test_data_t.treatment_proxy2.to(device)
        tempY0 = test_data_t.backdoor1.to(device)
        tempY1 = test_data_t.backdoor2.to(device)
        model_inputs_test = torch.cat((tempZ1, tempZ2, tempY0, tempY1),dim = 1)

        with torch.no_grad():
            pred = model(model_inputs_test)

        return pred
        
        
        
        
        
        
        
if __name__ == "__main__":        
    pass