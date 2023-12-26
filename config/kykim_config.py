import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dataloader import BaseDataloader
from dataset import Dataset
from model import BaseModel
from wrappers.train_wrapper import LossfunctionWrap, OptimizerWrap, SchedulerWrap
from wrappers.config_wrapper import TrainerConfig, DataConfig, ModelConfig, InferenceConfig


"""
kykim/electra-kor-base(모델) + AugmentationV2(데이터) Config 
"""

kykim_data_config = DataConfig(dataloader=BaseDataloader,
                                model_name='kykim/electra-kor-base',
                                dataset=Dataset,
                                batch_size=32,
                                shuffle=True,
                                train_path='../data/train_augmentV2.csv',
                                dev_path='../data/dev_spellcheck.csv',
                                test_path='../data/dev_spellcheck.csv',
                                predict_path='../data/test_spellcheck.csv'
                                )

kykim_model_config = ModelConfig(model=BaseModel,
                                 model_name='kykim/electra-kor-base',
                                 loss_func=LossfunctionWrap(loss=torch.nn.L1Loss),
                                 optimizer=OptimizerWrap(optimizer=torch.optim.Adam,
                                                         hyperparmeter={'lr':2*1e-5}),
                                 scheduler=SchedulerWrap(scheduler=CosineAnnealingWarmRestarts, 
                                                         hyperparmeter={'T_0':7, 'T_mult':2, 'eta_min':1e-6})
                                )

kykim_trainer_config = TrainerConfig(seed=42,
                                     epoch=23, # 23,
                                     save_path='../model/kykim-electra-kor-base.pt',
                                     precision=32, # 32
                                     callbacks= None,
                                     strategy='auto'
                                     ) 

kykim_inference_config = InferenceConfig(predict_path='../result/kykim-electra-kor-base.csv',
                                         model_path = '../model/kykim-electra-kor-base.pt',
                                         submission_path='../data/sample_submission.csv',
                                         ensemble_weight = 0.9217
                                         )