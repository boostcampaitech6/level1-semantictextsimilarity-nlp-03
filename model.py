from tqdm.auto import tqdm
import transformers
import torchmetrics
import pytorch_lightning as pl

from typing import Union
from wrappers.train_wrapper import LossfunctionWrap, OptimizerWrap, SchedulerWrap


class BaseModel(pl.LightningModule):
    """Baseline Code : pl.LightningModule을 상속받은 Model Class"""
    def __init__(self, model_name:str,loss_func:LossfunctionWrap, optimizer:OptimizerWrap, scheduler: Union[SchedulerWrap, None]):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name

        # 선언되지 않은 Loss, Optimizer, Scheduler 객체와 hyperparameter를 호출합니다.
        self.loss_object, self.loss_hyperparam = loss_func.loss, loss_func.hyperparmeter
        self.optimizer_object, self.optimizer_hyperparam = optimizer.optimizer, optimizer.hyperparmeter
        self.scheduler_object = None
        if scheduler:
            self.scheduler_object, self.scheduler_hyperparam = scheduler.scheduler, scheduler.hyperparmeter
        
        print("model_name : ", self.model_name)
        
        # Huggingface로 부터 사용할 모델을 호출합니다.
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
                                                pretrained_model_name_or_path=self.model_name, num_labels=1)

        # Loss 객체의 hyperparamter를 입력받아 Loss를 선언합니다.
        self.loss_func = self.loss_object(**self.loss_hyperparam)


    def forward(self, x):
        x = self.model(x)['logits']
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)
        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        return logits.squeeze()

    def configure_optimizers(self):
        # Optimizer, Scheduler 객체의 hyperparamter를 입력받아 Optimizer, Scheduler를 선언합니다.
        optimizer = self.optimizer_object(self.model.parameters(), **self.optimizer_hyperparam)
        
        # Scheduler가 있을 경우와 없을 경우에 따라 구분.
        if self.scheduler_object :
            scheduler = self.scheduler_object(optimizer, **self.scheduler_hyperparam)
            return [optimizer], [scheduler]
        else:
            return optimizer