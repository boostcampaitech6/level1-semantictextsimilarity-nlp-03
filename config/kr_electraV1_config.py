import torch
from dataloader import BaseDataloader
from dataset import Dataset
from model import BaseModel
from wrappers.train_wrapper import LossfunctionWrap, OptimizerWrap, SchedulerWrap
from wrappers.config_wrapper import TrainerConfig, DataConfig, ModelConfig, InferenceConfig


"""
snunlp/KR-ELECTRA-discriminator(모델) + AugmentationV1(데이터) Config 
"""

krelectraV1_data_config = DataConfig(dataloader=BaseDataloader,
                                model_name='snunlp/KR-ELECTRA-discriminator',
                                dataset=Dataset,
                                batch_size=32,
                                shuffle=True,
                                train_path='../data/train_augmentV1.csv',
                                dev_path='../data/dev.csv',
                                test_path='../data/dev.csv',
                                predict_path='../data/test.csv'
                                )

krelectraV1_model_config = ModelConfig(model=BaseModel,
                                 model_name='snunlp/KR-ELECTRA-discriminator',
                                 loss_func=LossfunctionWrap(loss=torch.nn.L1Loss),
                                 optimizer=OptimizerWrap(optimizer=torch.optim.Adam,
                                                         hyperparmeter={'lr':1e-5}),
                                 scheduler=None
                                )

krelectraV1_trainer_config = TrainerConfig(seed=0,
                                     epoch=15, # 15,
                                     save_path='../model/snunlp-KR-ELECTRA-discriminator-V1.pt',
                                     precision=32, # precision 32 맞는지 확인
                                     callbacks= None,
                                     strategy='auto')  

krelectraV1_inference_config = InferenceConfig(predict_path='../result/snunlp-KR-ELECTRA-discriminator-V1.csv',
                                         model_path = '../model/snunlp-KR-ELECTRA-discriminator-V1.pt',
                                         submission_path='../data/sample_submission.csv',
                                         ensemble_weight = 0.9166
                                         )

