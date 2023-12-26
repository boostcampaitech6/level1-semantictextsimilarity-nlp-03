from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from learner import Learner
from utils import seed_everything
from config.kykim_config import kykim_data_config, kykim_model_config, kykim_trainer_config, kykim_inference_config
from config.kr_electraV2_config import krelectraV2_data_config, krelectraV2_model_config, krelectraV2_trainer_config, krelectraV2_inference_config
from config.kr_electraV1_config import krelectraV1_data_config, krelectraV1_model_config, krelectraV1_trainer_config, krelectraV1_inference_config
from config.roberta_large_config import roberta_large_data_config, roberta_large_model_config, roberta_large_trainer_config, roberta_large_inference_config
from config.roberta_large_nnp_config import roberta_large_nnp_data_config, roberta_large_nnp_model_config, roberta_large_nnp_trainer_config, roberta_large_nnp_inference_config




if __name__ == '__main__':
    # (1) kykim/electra-kor-base(model) + AugmentationV2(data) 학습 및 저장 및 추론
    seed_everything(SEED=kykim_trainer_config.seed)
    kykim_learner = Learner(model_config=kykim_model_config,
                            data_config=kykim_data_config,
                            train_config=kykim_trainer_config,
                            inference_config=kykim_inference_config)

    kykim_learner.run_and_save()

    # (2) KR-ELECTRA-discriminator(model) + AugmentationV2(data)모델 학습 및 저장
    seed_everything(SEED=krelectraV2_trainer_config.seed)
    krelectraV2_learner = Learner(model_config=krelectraV2_model_config,
                                  data_config=krelectraV2_data_config,
                                  train_config=krelectraV2_trainer_config,
                                  inference_config=krelectraV2_inference_config)

    krelectraV2_learner.run_and_save()

    # (3) KR-ELECTRA-discriminator(model) + AugmentationV1(data) 모델 학습 및 저장
    seed_everything(SEED=krelectraV1_trainer_config.seed)
    krelectraV1_learner = Learner(model_config=krelectraV1_model_config,
                                  data_config=krelectraV1_data_config,
                                  train_config=krelectraV1_trainer_config,
                                  inference_config=krelectraV1_inference_config)

    krelectraV1_learner.run_and_save()

    # (4) klue/roberta-large(model) + AugmentationV2(data) 모델 학습 및 저장
    seed_everything(SEED=roberta_large_trainer_config.seed)
    roberta_large_learner = Learner(model_config=roberta_large_model_config,
                                  data_config=roberta_large_data_config,
                                  train_config=roberta_large_trainer_config,
                                  inference_config=roberta_large_inference_config)

    roberta_large_learner.run_and_save()

    # roberta_large : .ckpt/ (체크포인트 디렉토리)을 .pt (.pt 모델 파일)로 변환
    convert_zero_checkpoint_to_fp32_state_dict(checkpoint_dir = roberta_large_trainer_config.save_path[:-3] + 'ckpt/', 
                                               output_file = roberta_large_inference_config.model_path)
    
    # (5) klue/roberta-large(model) + AugmentationV3(data) 모델 학습 및 저장
    seed_everything(SEED=roberta_large_nnp_trainer_config.seed)
    roberta_large_nnp_learner = Learner(model_config=roberta_large_nnp_model_config,
                                  data_config=roberta_large_nnp_data_config,
                                  train_config=roberta_large_nnp_trainer_config,
                                  inference_config=roberta_large_nnp_inference_config)

    roberta_large_nnp_learner.run_and_save()

    # roberta_large : .ckpt/ (체크포인트 디렉토리)을 .pt (.pt 모델 파일)로 변환
    convert_zero_checkpoint_to_fp32_state_dict(checkpoint_dir = roberta_large_nnp_trainer_config.save_path[:-3] + 'ckpt/', 
                                               output_file = roberta_large_nnp_inference_config.model_path)


    

    