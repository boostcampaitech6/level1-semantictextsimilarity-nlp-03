class LossfunctionWrap():
    """Loss function 객체를 Config에서 선택하고 Model 내에서 함수를 정의(definition)하도록 도와주는 Wrapper class"""
    def __init__(self, loss, hyperparmeter={}):
        self.loss = loss
        self.hyperparmeter = hyperparmeter

class OptimizerWrap():
    """Optimizer 객체를 Config에서 선택하고 Model 내에서 함수를 정의(definition)하도록 도와주는 Wrapper class"""
    def __init__(self, optimizer, hyperparmeter={}):
        self.optimizer = optimizer
        self.hyperparmeter = hyperparmeter

class SchedulerWrap():
    """Scheduler 객체를 Config에서 선택하고 Model 내에서 함수를 정의(definition)하도록 도와주는 Wrapper class"""
    def __init__(self, scheduler, hyperparmeter={}):
        self.scheduler = scheduler
        self.hyperparmeter = hyperparmeter