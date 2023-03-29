import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, LSTM, Bidirectional
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.keras.layers import concatenate
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

def split_x(a, b):
    x=[]
    for i in range(len(a)-b-1):
        c = a[i:(i+b)]
        x.append(c)
    return np.array(x)

def rmse(a,b):
    return np.sqrt(mean_squared_error(a,b))


# 1. 데이터
path = "./_data/시험/"
path_save = "./_save/samsung/"
datasets_sam = pd.read_csv(path+'삼성전자 주가3.csv',index_col=0,encoding='cp949') 
datasets_hyun = pd.read_csv(path+'현대자동차2.csv',index_col=0,encoding='cp949') 

datasets_hyun =datasets_hyun[::-1]
datasets_sam = datasets_sam[::-1]
datasets_hyun =datasets_hyun[-180:]
datasets_sam = datasets_sam[-180:]

hyun_x = np.array(datasets_hyun.drop(['전일비','종가'],axis=1))
sam_x = np.array(datasets_sam.drop(['전일비','종가'],axis=1))
hyun_y = np.array(datasets_hyun['종가'])
sam_y = np.array(datasets_sam['종가'])

sam_x = np.char.replace(sam_x.astype(str), ',', '').astype(np.float64)
hyun_x = np.char.replace(hyun_x.astype(str), ',', '').astype(np.float64)
sam_y = np.char.replace(sam_y.astype(str), ',', '').astype(np.float64)
hyun_y = np.char.replace(hyun_y.astype(str), ',', '').astype(np.float64)

timesteps = 30

_, sam_x_test, _, sam_y_test, _, hyun_x_test, _, hyun_y_test = train_test_split(sam_x, sam_y, hyun_x, hyun_y, train_size=0.8, shuffle=False)
(sam_x_train,sam_y_train,hyun_x_train,hyun_y_train)=(sam_x, sam_y, hyun_x, hyun_y)

# sam_x_train, sam_x_test, sam_y_train, sam_y_test, hyun_x_train, hyun_x_test\
#     ,hyun_y_train, hyun_y_test = train_test_split(sam_x, sam_y, hyun_x, hyun_y,
#                                                   train_size=0.9, shuffle=False)

scaler = MinMaxScaler()
sam_x_train = scaler.fit_transform(sam_x_train)
sam_x_test = scaler.transform(sam_x_test)
hyun_x_train = scaler.transform(hyun_x_train)
hyun_x_test = scaler.transform(hyun_x_test)

sam_x_train_split = split_x(sam_x_train, timesteps)
sam_x_test_split = split_x(sam_x_test, timesteps)
hyun_x_train_split = split_x(hyun_x_train, timesteps)
hyun_x_test_split = split_x(hyun_x_test, timesteps)

sam_y_train_split = sam_y_train[timesteps+1:]
sam_y_test_split = sam_y_test[timesteps+1:]
hyun_y_train_split = hyun_y_train[timesteps+1:]
hyun_y_test_split = hyun_y_test[timesteps+1:]

# 2. 모델 구성
#2-1. 모델 1
input1 = Input(shape=(timesteps,14))
lstm_BI01 = Bidirectional(LSTM(128,name='s1'))(input1)
dense1 = Dense(64,activation='relu',name='s2')(lstm_BI01)
dense2 = Dense(64,name='s3')(dense1)
dense3 = Dense(64,name='s4')(dense2)
output1 = Dense(64,activation='relu',name='os')(dense3)

#2-2. 모델 2
input2 = Input(shape=(timesteps,14))
lstm_BI11 = Bidirectional(LSTM(128,name='h1'))(input2)
dense12 = Dense(64,activation='relu',name='h2')(lstm_BI11)
dense13 = Dense(64,name='h3')(dense12)
dense14 = Dense(64,name='h4')(dense13)
output2 = Dense(64,activation='relu',name='oh')(dense14)

# 2-3. 머지
merge1 = concatenate([output1,output2],name='mg1')
merge2 = Dense(64,name='mg2')(merge1)
merge3 = Dense(64,activation='relu',name='mg3')(merge2)
hidden_output = Dense(64,name='last')(merge3)

#2-4. 분기 모델 1
bungi11= Dense(64,activation='relu',name='result01')(hidden_output)
bungi12= Dense(64,activation='relu',name='result02')(bungi11)
bungi13= Dense(32,name='result13')(bungi12)
last_output01= Dense(1,activation='relu',name='last_output01')(bungi12)

#2-5. 분기 모델 2
bungi21= Dense(128,activation='relu',name='result11')(hidden_output)
bungi22= Dense(64,activation='relu',name='result12')(bungi21)
bungi23= Dense(32,name='result13')(bungi22)
last_output02= Dense(1,name='last_output02')(bungi23)
model = Model(inputs=[input1,input2], outputs=[last_output01,last_output02])

# 3. 컴파일 훈련 
model.compile(loss='mse',optimizer='adam')
es = EarlyStopping(
    monitor='loss',
    patience=50,
    mode='min',
    verbose=1,
    restore_best_weights=True
)
mcp=ModelCheckpoint(monitor='loss',mode='min',verbose=1, save_best_only=True,
                    filepath='./_save/samsung/keras53_samsung4_kjy_save02.hdf5'
                    )
model.fit(
    [sam_x_train_split, hyun_x_train_split],
    [sam_y_train_split, hyun_y_train_split],
    epochs=150, batch_size=20,
    validation_split=0.2,
    callbacks=[es, mcp]
)

# 4. 평가 예측 
loss=model.evaluate([sam_x_test_split, hyun_x_test_split], [sam_y_test_split, hyun_y_test_split])
print('loss : ',loss)

sam_x_pred =np.array(sam_x_test[-timesteps:]).reshape(1, timesteps, 14)
hyun_x_pred =np.array(hyun_x_test[-timesteps:]).reshape(1, timesteps, 14)
result = model.predict([sam_x_pred, hyun_x_pred])

print(np.round(result[1],2))
