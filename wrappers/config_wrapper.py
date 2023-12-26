from typing import Union

from dataloader import BaseDataloader
from dataset import Dataset
from model import BaseModel
from wrappers.train_wrapper import LossfunctionWrap, OptimizerWrap, SchedulerWrap
from typing import Union, List


class DataConfig():
    """Dataset, DataLoader와 관련된 변수들을 관리하는 config class"""
    def __init__(self, dataloader:BaseDataloader, model_name:str, dataset:Dataset, batch_size:int, shuffle:bool,
                 train_path:str, dev_path:str, test_path:str, predict_path:str):
        self.dataloader:BaseDataloader = dataloader
        self.model_name:str = model_name
        self.dataset:Dataset = dataset
        self.batch_size:int = batch_size
        self.shuffle:bool = shuffle
        self.train_path:str = train_path
        self.dev_path:str = dev_path
        self.test_path:str = test_path
        self.predict_path:str = predict_path


class ModelConfig():
    """Model과 관련된 변수들을 관리하는 config class"""
    def __init__(self, model:BaseModel, model_name:str, loss_func:LossfunctionWrap, 
                 optimizer:OptimizerWrap, scheduler:Union[SchedulerWrap, None]=None):
        
        self.model:BaseModel = model
        self.model_name:str = model_name
        self.loss_func:LossfunctionWrap = loss_func
        self.optimizer:OptimizerWrap = optimizer
        self.scheduler:SchedulerWrap = scheduler


class TrainerConfig():
    """Pytorch-lightning Trainer와 관련된 변수들을 관리하는 config class"""
    def __init__(self, seed:int, epoch:str, save_path:str, precision:int, 
                        callbacks:Union[List, None]=None, strategy:str='auto'):
        self.seed:int = seed
        self.epoch:str = epoch
        self.save_path:str = save_path
        self.precision:int = precision
        self.callbacks:Union[List, None] = callbacks
        self.strategy:str = strategy

class InferenceConfig():
    """모델 추론과 관련된 변수들을 관리하는 config class"""
    def __init__(self, predict_path:str, model_path:str, submission_path:str, ensemble_weight:float):
        self.predict_path:str = predict_path
        self.model_path:str = model_path
        self.submission_path:str = submission_path
        self.ensemble_weight:float = ensemble_weight

