import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloader import BaseDataloader
from dataset import Dataset
from model import BaseModel
from wrappers.train_wrapper import LossfunctionWrap, OptimizerWrap, SchedulerWrap
from wrappers.config_wrapper import TrainerConfig, DataConfig, ModelConfig, InferenceConfig


"""
klue/roberta-large(모델) + AugmentationV3(데이터) Config 
"""


roberta_large_nnp_data_config = DataConfig(dataloader=BaseDataloader,
                                model_name='klue/roberta-large',
                                dataset=Dataset,
                                batch_size=16,
                                shuffle=True,
                                train_path='../data/train_augmentV3.csv',
                                dev_path='../data/dev_spellcheck.csv',
                                test_path='../data/dev_spellcheck.csv',
                                predict_path='../data/test_spellcheck.csv'
                                )

roberta_large_nnp_model_config = ModelConfig(model=BaseModel,
                                 model_name='klue/roberta-large',
                                 loss_func=LossfunctionWrap(loss=torch.nn.L1Loss),
                                 optimizer=OptimizerWrap(optimizer=torch.optim.AdamW,
                                                         hyperparmeter={'lr':1e-5}),
                                 scheduler=SchedulerWrap(scheduler=CosineAnnealingLR, 
                                                         hyperparmeter={'T_max':50})
                                )

roberta_large_nnp_trainer_config = TrainerConfig(seed=0,
                                     epoch=2, # 2,
                                     save_path='../model/klue-roberta-large-nnp.pth',
                                     precision="16-mixed",
                                     callbacks=[ModelCheckpoint(monitor="val_pearson",
                                                                dirpath='../model',
                                                                filename='klue-roberta-large-nnp',
                                                                mode='max',
                                                                save_top_k=1
                                                                )],
                                     strategy='deepspeed_stage_2') 


roberta_large_nnp_inference_config = InferenceConfig(predict_path='../result/klue-roberta-large-nnp.csv',
                                                    model_path = '../model/klue-roberta-large-nnp.pt',
                                                    submission_path='../data/sample_submission.csv',
                                                    ensemble_weight = 0.9166
                                                    )