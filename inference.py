import argparse
from learner import Learner
from utils import seed_everything
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from config.kykim_config import kykim_data_config, kykim_model_config, kykim_trainer_config, kykim_inference_config
from config.kr_electraV2_config import krelectraV2_data_config, krelectraV2_model_config, krelectraV2_trainer_config, krelectraV2_inference_config
from config.kr_electraV1_config import krelectraV1_data_config, krelectraV1_model_config, krelectraV1_trainer_config, krelectraV1_inference_config
from config.roberta_large_config import roberta_large_data_config, roberta_large_model_config, roberta_large_trainer_config, roberta_large_inference_config
from config.roberta_large_nnp_config import roberta_large_nnp_data_config, roberta_large_nnp_model_config, roberta_large_nnp_trainer_config, roberta_large_nnp_inference_config

from utils import ensemble



if __name__ == '__main__':

    # 터미널 실행시 --inference=True (default)는 학습한 모델을 입력받아 추론을, False는 앙상블을 진행합니다.
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='inference', type=str)
    args = parser.parse_args()

    assert args.mode in ['inference', 'ensemble'], "--mode should be 'inference' or 'ensemble'"
    # 추론
    if args.mode == 'inference':
        
        # (1) kykim/electra-kor-base(model) + AugmentationV2(data)모델 추론
        seed_everything(SEED=kykim_trainer_config.seed)
        kykim_learner = Learner(model_config=kykim_model_config,
                                data_config=kykim_data_config,
                                train_config=kykim_trainer_config,
                                inference_config=kykim_inference_config)

        kykim_learner.predict()

        # (2) KR-ELECTRA-discriminator(model) + AugmentationV2(data)모델 추론
        seed_everything(SEED=krelectraV2_trainer_config.seed)
        krelectraV2_learner = Learner(model_config=krelectraV2_model_config,
                                    data_config=krelectraV2_data_config,
                                    train_config=krelectraV2_trainer_config,
                                    inference_config=krelectraV2_inference_config)

        krelectraV2_learner.predict()

        # (3) KR-ELECTRA-discriminator(model) + AugmentationV1(data)모델 추론
        seed_everything(SEED=krelectraV1_trainer_config.seed)
        krelectraV1_learner = Learner(model_config=krelectraV1_model_config,
                                    data_config=krelectraV1_data_config,
                                    train_config=krelectraV1_trainer_config,
                                    inference_config=krelectraV1_inference_config)

        krelectraV1_learner.predict()

        # (4) klue/roberta-large(model) + AugmentationV2(data)모델 추론
        seed_everything(SEED=roberta_large_trainer_config.seed)
        roberta_large_learner = Learner(model_config=roberta_large_model_config,
                                    data_config=roberta_large_data_config,
                                    train_config=roberta_large_trainer_config,
                                    inference_config=roberta_large_inference_config)

        roberta_large_learner.predict()

        # (5) klue/roberta-large(model) + AugmentationV2 + nnp(data)모델 추론
        seed_everything(SEED=roberta_large_nnp_trainer_config.seed)
        roberta_large_nnp_learner = Learner(model_config=roberta_large_nnp_model_config,
                                    data_config=roberta_large_nnp_data_config,
                                    train_config=roberta_large_nnp_trainer_config,
                                    inference_config=roberta_large_nnp_inference_config)

        roberta_large_learner.predict()


    # Ensemble
    elif args.mode == 'ensemble':
        result_path_list = [kykim_inference_config.predict_path, 
                            krelectraV2_inference_config.predict_path, 
                            krelectraV1_inference_config.predict_path,
                            roberta_large_inference_config.predict_path,
                            roberta_large_nnp_inference_config.predict_path]
        
        score_list = [kykim_inference_config.ensemble_weight, 
                      krelectraV2_inference_config.ensemble_weight, 
                      krelectraV1_inference_config.ensemble_weight,
                      roberta_large_inference_config.ensemble_weight,
                      roberta_large_nnp_inference_config.ensemble_weight]
        
        postprocessing_list = [True, False, False, False, False]

        # 모델 앙상블
        ensemble(result_path_list = result_path_list, 
                 score_list = score_list, 
                 postprocessing_list = postprocessing_list,
                 save_path = '../result/ensemle.csv')
        
        
