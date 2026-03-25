from typing import NamedTuple, Optional
import numpy as np
import torch
from sklearn.model_selection import train_test_split

'''
Data classes for MMR datasets
'''

class MMRTrainDataSet_h(NamedTuple):
    treatment: np.ndarray
    treatment_proxy: np.ndarray
    outcome_proxy: np.ndarray
    outcome: np.ndarray
    backdoor: np.ndarray

class MMRTestDataSet_h(NamedTuple):
    treatment: np.ndarray
    treatment_proxy: np.ndarray
    outcome_proxy: np.ndarray
    outcome: np.ndarray
    backdoor: np.ndarray

class MMRTrainDataSet_q(NamedTuple):
    treatment: np.ndarray
    treatment_target:np.ndarray
    treatment_proxy: np.ndarray
    outcome_proxy: np.ndarray
    outcome: np.ndarray
    backdoor: np.ndarray
    
class MMRTrainDataSet_q_2stage(NamedTuple):
    treatment1: np.ndarray
    treatment_target1:np.ndarray
    treatment_proxy1: np.ndarray
    outcome_proxy1: np.ndarray
    outcome1: np.ndarray
    backdoor1: np.ndarray

    treatment2: np.ndarray
    treatment_target2:np.ndarray
    treatment_proxy2: np.ndarray
    outcome_proxy2: np.ndarray
    outcome2: np.ndarray
    backdoor2: np.ndarray

class MMRTestDataSet_q(NamedTuple):
    treatment: np.ndarray
    treatment_proxy: np.ndarray
    outcome_proxy: np.ndarray
    outcome: np.ndarray
    backdoor: np.ndarray
    structural: np.ndarray

class MMRTestDataSet_q_2stage(NamedTuple):
    treatment1: np.ndarray
    treatment_proxy1: np.ndarray
    outcome_proxy1: np.ndarray
    outcome1: np.ndarray
    backdoor1: np.ndarray
    structural1: np.ndarray
    
    treatment2: np.ndarray
    treatment_proxy2: np.ndarray
    outcome_proxy2: np.ndarray
    outcome2: np.ndarray
    backdoor2: np.ndarray   



##--Torch versions of the datasets--##

class MMRTrainDataSetTorch_h(NamedTuple):
    treatment: torch.Tensor
    treatment_proxy: torch.Tensor
    outcome_proxy: torch.Tensor
    outcome: torch.Tensor
    backdoor: torch.Tensor

    @classmethod
    def from_numpy(cls, train_data: MMRTrainDataSet_h):
        backdoor = None
        if train_data.backdoor is not None:
            backdoor = torch.tensor(train_data.backdoor, dtype=torch.float32)
        return MMRTrainDataSetTorch_h(treatment=torch.tensor(train_data.treatment, dtype=torch.float32),
                                   treatment_proxy=torch.tensor(train_data.treatment_proxy, dtype=torch.float32),
                                   outcome_proxy=torch.tensor(train_data.outcome_proxy, dtype=torch.float32),
                                   backdoor=backdoor,
                                   outcome=torch.tensor(train_data.outcome, dtype=torch.float32))

    def to_gpu(self):
        backdoor = None
        if self.backdoor is not None:
            backdoor = self.backdoor.cuda()
        return MMRTrainDataSetTorch_h(treatment=self.treatment.cuda(),
                                   treatment_proxy=self.treatment_proxy.cuda(),
                                   outcome_proxy=self.outcome_proxy.cuda(),
                                   backdoor=backdoor,
                                   outcome=self.outcome.cuda())

class MMRTestDataSetTorch_h(NamedTuple):
    treatment: torch.Tensor
    outcome_proxy: torch.Tensor
    backdoor: torch.Tensor
    structural: Optional[torch.Tensor]

    @classmethod
    def from_numpy(cls, test_data: MMRTestDataSet_h):
        structural = None
        if hasattr(test_data, 'structural'):
            structural = torch.tensor(test_data.structural, dtype=torch.float32)
        return MMRTestDataSetTorch_h(treatment=torch.tensor(test_data.treatment, dtype=torch.float32),
                                   outcome_proxy=torch.tensor(test_data.outcome_proxy, dtype=torch.float32),
                                   backdoor=torch.tensor(test_data.backdoor, dtype=torch.float32),
                                   structural=structural)

    def to_gpu(self):
        structural = None
        if self.structural is not None:
            structural = self.structural.cuda()
        return MMRTestDataSetTorch_h(treatment=self.treatment.cuda(),
                                   outcome_proxy=self.outcome_proxy.cuda(),
                                   backdoor=self.backdoor.cuda(),
                                   structural=structural)
        
class MMRTrainDataSetTorch_q(NamedTuple):
    treatment: torch.Tensor
    treatment_target:torch.Tensor
    treatment_proxy: torch.Tensor
    outcome_proxy: torch.Tensor
    outcome: torch.Tensor
    backdoor: torch.Tensor

    @classmethod
    def from_numpy(cls, train_data: MMRTrainDataSet_q):
        if hasattr(train_data, 'treatment_target'):
            A_target = train_data.treatment_target
        else:
            A_target = np.zeros(train_data.treatment.shape)
        return MMRTrainDataSetTorch_q(treatment=torch.tensor(train_data.treatment, dtype=torch.float32),
                                   treatment_target=torch.tensor(A_target, dtype=torch.float32),
                                   treatment_proxy=torch.tensor(train_data.treatment_proxy, dtype=torch.float32),
                                   outcome_proxy=torch.tensor(train_data.outcome_proxy, dtype=torch.float32),
                                   backdoor=torch.tensor(train_data.backdoor, dtype=torch.float32),
                                   outcome=torch.tensor(train_data.outcome, dtype=torch.float32))

    def to_gpu(self):
        backdoor = None
        if self.backdoor is not None:
            backdoor = self.backdoor.cuda()
        return MMRTrainDataSetTorch_q(treatment=self.treatment.cuda(),
                                   treatment_target=self.treatment_target.cuda(),
                                   treatment_proxy=self.treatment_proxy.cuda(),
                                   outcome_proxy=self.outcome_proxy.cuda(),
                                   backdoor=backdoor,
                                   outcome=self.outcome.cuda())

class MMRTrainDataSetTorch_q_2stage(NamedTuple):
    treatment1: torch.Tensor
    treatment_target1:torch.Tensor
    treatment_proxy1: torch.Tensor
    outcome_proxy1: torch.Tensor
    outcome1: torch.Tensor
    backdoor1: torch.Tensor
    
    treatment2: torch.Tensor
    treatment_target2:torch.Tensor
    treatment_proxy2: torch.Tensor
    outcome_proxy2: torch.Tensor
    outcome2: torch.Tensor
    backdoor2: torch.Tensor
    
    @classmethod
    def from_numpy(cls, train_data: MMRTrainDataSet_q_2stage):
        if hasattr(train_data, 'treatment_target1'):
            A_target1 = train_data.treatment_target1
        else:
            A_target1 = np.zeros(train_data.treatment1.shape)
        if hasattr(train_data, 'treatment_target2'):
            A_target2 = train_data.treatment_target2
        else:
            A_target2 = np.zeros(train_data.treatment2.shape)
        return MMRTrainDataSetTorch_q_2stage(
            treatment1=torch.tensor(train_data.treatment1, dtype=torch.float32),
            treatment_target1=torch.tensor(A_target1, dtype=torch.float32),
            treatment_proxy1=torch.tensor(train_data.treatment_proxy1, dtype=torch.float32),
            outcome_proxy1=torch.tensor(train_data.outcome_proxy1, dtype=torch.float32),
            backdoor1=torch.tensor(train_data.backdoor1, dtype=torch.float32),
            outcome1=torch.tensor(train_data.outcome1, dtype=torch.float32),
            treatment2=torch.tensor(train_data.treatment2, dtype=torch.float32),
            treatment_target2=torch.tensor(A_target2, dtype=torch.float32),
            treatment_proxy2=torch.tensor(train_data.treatment_proxy2, dtype=torch.float32),
            outcome_proxy2=torch.tensor(train_data.outcome_proxy2, dtype=torch.float32),
            backdoor2=torch.tensor(train_data.backdoor2, dtype=torch.float32),
            outcome2=torch.tensor(train_data.outcome2, dtype=torch.float32))
            
    def to_gpu(self):
        return MMRTrainDataSetTorch_q_2stage(
            treatment1=self.treatment1.cuda(),
            treatment_target1=self.treatment_target1.cuda(),
            treatment_proxy1=self.treatment_proxy1.cuda(),
            outcome_proxy1=self.outcome_proxy1.cuda(),
            backdoor1=self.backdoor1.cuda(),
            outcome1=self.outcome1.cuda(),
            treatment2=self.treatment2.cuda(),
            treatment_target2=self.treatment_target2.cuda(),
            treatment_proxy2=self.treatment_proxy2.cuda(),
            outcome_proxy2=self.outcome_proxy2.cuda(),
            backdoor2=self.backdoor2.cuda(),
            outcome2=self.outcome2.cuda()
        )

class MMRTestDataSetTorch_q(NamedTuple):
    treatment: torch.Tensor
    treatment_proxy: torch.Tensor
    outcome: torch.Tensor
    outcome_proxy: torch.Tensor     #new 
    backdoor: torch.Tensor
    structural: Optional[torch.Tensor]

    @classmethod
    def from_numpy(cls, test_data: MMRTestDataSet_q):
        structural = None
        if hasattr(test_data, 'structural'):
            structural = torch.tensor(test_data.structural, dtype=torch.float32)
        return MMRTestDataSetTorch_q(treatment=torch.tensor(test_data.treatment, dtype=torch.float32),
                                   treatment_proxy=torch.tensor(test_data.treatment_proxy, dtype=torch.float32),
                                   outcome=torch.tensor(test_data.outcome, dtype=torch.float32),
                                   outcome_proxy=torch.tensor(test_data.outcome_proxy, dtype=torch.float32),
                                   backdoor=torch.tensor(test_data.backdoor, dtype=torch.float32),
                                   structural=structural)

    def to_gpu(self):
        structural = None
        if self.structural is not None:
            structural = self.structural.cuda()
        return MMRTestDataSetTorch_q(treatment=self.treatment.cuda(),
                                   treatment_proxy=self.treatment_proxy.cuda(),
                                   outcome=self.outcome.cuda(),
                                   outcome_proxy=self.outcome_proxy.cuda(),
                                   backdoor=self.backdoor.cuda(),
                                   structural=structural)

class MMRTestDataSetTorch_q_2stage(NamedTuple):
    treatment1: torch.Tensor
    treatment_proxy1: torch.Tensor
    outcome1: torch.Tensor
    outcome_proxy1: torch.Tensor     #new
    backdoor1: torch.Tensor
    structural1: Optional[torch.Tensor]
    
    treatment2: torch.Tensor
    treatment_proxy2: torch.Tensor
    outcome2: torch.Tensor
    outcome_proxy2: torch.Tensor     #new
    backdoor2: torch.Tensor          #=outcome1
    
    @classmethod
    def from_numpy(cls, test_data: MMRTestDataSet_q_2stage):
        structural1 = None
        if hasattr(test_data, 'structural1'):
            structural1 = torch.tensor(test_data.structural1, dtype=torch.float32)
        return MMRTestDataSetTorch_q_2stage(
            treatment1=torch.tensor(test_data.treatment1, dtype=torch.float32),
            treatment_proxy1=torch.tensor(test_data.treatment_proxy1, dtype=torch.float32),
            outcome1=torch.tensor(test_data.outcome1, dtype=torch.float32),
            outcome_proxy1=torch.tensor(test_data.outcome_proxy1, dtype=torch.float32),
            backdoor1=torch.tensor(test_data.backdoor1, dtype=torch.float32),
            structural1=torch.tensor(test_data.structural1,dtype=torch.float32),
            treatment2=torch.tensor(test_data.treatment2, dtype=torch.float32),
            treatment_proxy2=torch.tensor(test_data.treatment_proxy2, dtype=torch.float32),
            outcome2=torch.tensor(test_data.outcome2, dtype=torch.float32),
            outcome_proxy2=torch.tensor(test_data.outcome_proxy2, dtype=torch.float32),
            backdoor2=torch.tensor(test_data.backdoor2, dtype=torch.float32)
        )




def split_train_data(train_data: MMRTrainDataSet_h, split_ratio=0.5):
    if split_ratio < 0.0:
        return train_data, train_data

    n_data = train_data[0].shape[0]
    idx_train_1st, idx_train_2nd = train_test_split(np.arange(n_data), train_size=split_ratio)

    def get_data(data, idx):
        return data[idx] if data is not None else None

    train_1st_data = MMRTrainDataSet_h(*[get_data(data, idx_train_1st) for data in train_data])
    train_2nd_data = MMRTrainDataSet_h(*[get_data(data, idx_train_2nd) for data in train_data])
    return train_1st_data, train_2nd_data
