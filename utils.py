import random
import numpy as np
import pandas as pd
import torch
from typing import List


def seed_everything(SEED=42):
    """
    시드 고정
    """
    deterministic = True
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def postprocessing(df:pd.DataFrame) -> pd.DataFrame:
    """
    후처리 함수 : 예측값에 단순 -0.1 뺄셈
    """
    df['target'] = df['target'].apply(lambda x : max(0, x-0.1))
    return df



def ensemble(result_path_list:List[pd.DataFrame], score_list:List[float], 
                postprocessing_list:List[bool], save_path='../result/ensemble.csv') -> pd.DataFrame:
    """
    점수 가중 평균 Ensemble
    """
    df_submission, weight_sum = None, 0 
    for i, (path, weight, pp) in enumerate(zip(result_path_list, score_list, postprocessing_list)):
        df_now = pd.read_csv(path)
        # 후처리를 진행
        if pp:
            df_now = postprocessing(df_now)
        
        # i == 0에서 제출 파일 생성 / 점수 가중 합
        if i == 0:
            df_submission = pd.read_csv(path)
            df_submission['target'] = weight * df_now['target']
        else:
            df_submission['target'] += weight * df_now['target']
        
        weight_sum += weight
    
    # 점수 가중 평균
    df_submission['target'] /= weight_sum
    df_submission.to_csv(save_path)

     