'''
추가해야하는 사항 
1. 파생 피처 / 예를 들면 전일비 등 
   그리고 전에 했던 데이터 셋 참고해서 파생피처 만들기
2. x_split -> 이번에는 선생님께서 전에 주신 함수로 사용가능


전체적인 코드 작성 방법
1) 각 종목따라 데이터 따로 모으기
2) 그 후, 모은걸로 시계열 모델로 예측하고 
3) 랭크를 먹이는 것 

첫 번째 전처리 경우
0) 결측치 알아보기  
1) 파생피처 
2) 시계열 모델이 알아듣도록 일자에 대한 데이터 변경 (이건 알아봐야함)
3) x_split 해주기 
4) split한 것으로 y 맞춰주기 
5) 이상치에 대해서는 할지 말지 고민 

두 번째 모델 
1) 단일 lstm 
2) 두 개의 lstm
3) 세 개의 lstm
4) BiLstm - 단일 두개, 세개 그리고 비교
5) 단변량 소타 모델 사용 
6) 다변량 소타 모델 사용

세 번째 컴파일 훈련 
1) 얼리스탑핑 
2) 러닝레이트 조절 하는 거 사용 

네 번째 평가 예측  
1) y_pred 하고 나서
2) 각 종목에 대해서 랭크를 먹이기

다섯 번째 제출 
이미 코드 작성되어있음 

'''

import pandas as pd
import numpy as np
import random
import os
from time import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,BatchNormalization,Bidirectional,Dense,ReLU
from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA

import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Seed 고정


def LSTM02():
    model = Sequential()
    model.add(LSTM(128,input_shape=()))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(LSTM(64,return_sequences=True))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(1))
    return model




# 1. 데이터

train_csv = pd.read_csv('./_data/joosick/train.csv')
train_csv = train_csv.drop(['종목명'],axis=-1)

# train 데이터에 존재하는 독립적인 종목코드 추출
unique_codes = train_csv['종목코드'].unique()
print(f'unique_codes : {unique_codes}')

# 각 종목코드에 대해서 모델 학습 및 추론 반복
for code in tqdm(unique_codes):
    
    # 전처리 
    train_close = train_csv[train_csv['종목코드'] == code][['일자','거래량','시가','고가','저가', '종가']]

    train_close['일자'] = pd.to_datetime(train_close['일자'], format='%Y%m%d')
    train_close.set_index('일자', inplace=True)
    tc = train_close['종가']
    print(f'tc.shape : {tc.shape}') #(494,)
    
    # x, y 나누기 


    # 2. 모델 선언, 학습 및 추론
    # model = ARIMA(tc, order=(2, 1, 2))
    model = LSTM02()

    #3. 모델 
    model.compile(loss='mse',optimizer='adam')
    from tensorflow.python.keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor='loss',patience=45,mode='min',verbose=1,restore_best_weights=True)
    model.fit(x,y,epochs=2000,batch_size=32,callbacks=[es])

    # predictions = model_fit.forecast(steps=15) # 향후 15개의 거래일에 대해서 예측

    # 4. 예측, 결과 
    # 최종 수익률 계산
    # final_return = (predictions.iloc[-1] - predictions.iloc[0]) / predictions.iloc[0]
    
# 5. 결과 저장

results_df = results_df.append({'종목코드': code, 'final_return': final_return}, ignore_index=True)

results_df['순위'] = results_df['final_return'].rank(method='first', ascending=False).astype('int') # 각 순위를 중복없이 생성
sample_submission = pd.read_csv('./_data/joosick/sample_submission.csv')
baseline_submission = sample_submission[['종목코드']].merge(results_df[['종목코드', '순위']], on='종목코드', how='left')
baseline_submission.to_csv('baseline_submission.csv', index=False)
