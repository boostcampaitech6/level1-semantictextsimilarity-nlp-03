import pandas as pd
import torch
import pytorch_lightning as pl
from wrappers.config_wrapper import TrainerConfig, DataConfig, ModelConfig, InferenceConfig

class Learner():
    """
    model_config, data_config, train_config, inference_config 를 입력받아서 모델을 훈련 및 저장하는 클래스.
    
    model_config : Model과 관련된 변수들을 관리하는 config
    data_config : Dataset, DataLoader와 관련된 변수들을 관리하는 config
    train_config : Pytorch-lightning Trainer와 관련된 변수들을 관리하는 config
    inference_config : 모델 추론과 관련된 변수들을 관리하는 config

    """
    def __init__(self, model_config:ModelConfig, data_config:DataConfig, 
                        train_config:TrainerConfig, inference_config:InferenceConfig):
        
        self.DATACONFIG = data_config
        self.MODELCONFIG = model_config
        self.TRAINCONFIG = train_config
        self.INFERCONFIG = inference_config

        # Dataloader 객체 선언 및 인스턴스 생성.
        dataloader_object = self.DATACONFIG.dataloader
        self.dataloader = dataloader_object(model_name = self.DATACONFIG.model_name,
                                            dataset = self.DATACONFIG.dataset,
                                            batch_size = self.DATACONFIG.batch_size,
                                            shuffle = self.DATACONFIG.shuffle,
                                            train_path = self.DATACONFIG.train_path,
                                            dev_path = self.DATACONFIG.dev_path,
                                            test_path = self.DATACONFIG.test_path,
                                            predict_path = self.DATACONFIG.predict_path)

        # Model 객체 선언 및 인스턴스 생성
        self.model_object = self.MODELCONFIG.model
        self.model = self.model_object(model_name = self.MODELCONFIG.model_name,
                                  loss_func=self.MODELCONFIG.loss_func,
                                  optimizer=self.MODELCONFIG.optimizer,
                                  scheduler=self.MODELCONFIG.scheduler)
        # Traniner 인스턴스 생성
        self.trainer = pl.Trainer(accelerator="gpu", 
                                devices=1,
                                max_epochs=self.TRAINCONFIG.epoch,
                                precision=self.TRAINCONFIG.precision,
                                strategy=self.TRAINCONFIG.strategy,
                                callbacks=self.TRAINCONFIG.callbacks,
                                log_every_n_steps=1)
            

    def run_and_save(self):
        # Train & Test with dev set
        self.trainer.fit(model=self.model, datamodule=self.dataloader)
        self.trainer.test(model=self.model, datamodule=self.dataloader)
        
        # 학습이 완료된 모델을 저장. roberta-large는 deepspeed 훈련을 통해 checkpoint로 저장
        if self.MODELCONFIG.model_name != 'klue/roberta-large':
            torch.save(self.model, self.TRAINCONFIG.save_path)


    def predict(self):
        # 모델 로드 및 예측
        if self.MODELCONFIG.model_name == 'klue/roberta-large':
            self.model_object.load_from_checkpoint(self.INFERCONFIG.model_path)
            predictions = self.trainer.predict(model=self.model, datamodule=self.dataloader) 
        
        else:
            model = torch.load(self.INFERCONFIG.model_path)
            predictions = self.trainer.predict(model=model, datamodule=self.dataloader)
        
        # 예측된 결과를 형식에 맞게 반올림, Clamping(0, 5)하여 준비합니다.
        predictions = list(max(0, min(round(float(i), 1), 5)) for i in torch.cat(predictions))
        
        # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
        output = pd.read_csv(self.INFERCONFIG.submission_path)
        output['target'] = predictions
        output.to_csv(self.INFERCONFIG.predict_path, index=False)