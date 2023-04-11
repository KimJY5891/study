import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers import Dense,Input, Conv1D,Flatten ,Dropout,LSTM,Bidirectional
from sklearn.metrics import r2_score, accuracy_score,mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
def split_x(dataset, timesteps) :
    x_list =[]
    for i in range(len(dataset)-timesteps) :
        subset = dataset[i:(i+timesteps)]
        x_list.append(subset)
    return np.array(x_list)
def rmse(y_test,y_predict) :
    return np.sqrt(mean_squared_error(y_test,y_predict))


# 1. 데이터

path = "./_data/시험/"
path_save = "./_save/samsung/"
datasets_sam = pd.read_csv(path+'삼성전자 주가3.csv',index_col=0,encoding='cp949')
print(datasets_sam.shape) #(3260, 16)
datasets_hyun = pd.read_csv(path+'현대자동차2.csv',index_col=0,encoding='cp949')
print(datasets_hyun.shape) #(3140, 16)
print(datasets_sam.columns)
# Index(['시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)',
#        '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'],
#       dtype='object')
print(datasets_hyun.columns)
# Index(['시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)',
#        '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'],
#       dtype='object')
print(datasets_sam.info())
print(datasets_sam.describe())
print(datasets_hyun.info())
print(datasets_hyun.describe())
print("datasets_sam['전일비']의 라벨 값 :",np.unique(datasets_sam['전일비'])) #[' ' '▲' '▼']
print("datasets_hyun['전일비']의 라벨 값 :",np.unique(datasets_hyun['전일비'])) #[' ' '▲' '▼']

datasets_hyun =datasets_hyun[::-1]
datasets_sam = datasets_sam[::-1]
datasets_hyun =datasets_hyun[-180:]
datasets_sam = datasets_sam[-180:]
print(datasets_sam.head())
print(datasets_hyun.head())
print(datasets_sam.shape) #(180, 16)
print(datasets_hyun.shape) #(180, 16)

# 데이터에 대한 레이블 인코딩 수행
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() 
le.fit(datasets_sam['전일비'])
aaa = le.transform(datasets_sam['전일비']) # 0과 1로 변화
print(aaa.shape)
print(type(aaa))
datasets_sam['전일비'] = aaa
print(" datasets_sam['전일비']의 라벨 값 :",np.unique(datasets_sam['전일비'])) #[' ' '▲' '▼']

le.fit(datasets_hyun['전일비'])
bbb = le.transform(datasets_hyun['전일비']) # 0과 1로 변화
print(bbb.shape)
print(type(bbb))
datasets_hyun['전일비'] = bbb
print(" datasets_sam['전일비']의 라벨 값 :",np.unique(datasets_hyun['전일비'])) #[' ' '▲' '▼']

datasets_sam = datasets_sam.astype(str).apply(lambda x: x.str.replace(',', '')).astype(np.float64)
datasets_hyun = datasets_hyun.astype(str).apply(lambda x: x.str.replace(',', '')).astype(np.float64)

hyun_x = datasets_hyun.drop(['종가'],axis=1)
sam_x = datasets_sam.drop(['종가'],axis=1)
hyun_y = datasets_hyun['종가']
sam_y = datasets_sam['종가']

print('hyun_x.shape : ',hyun_x.shape,'hyun_y.shpe : ',hyun_y.shape) # hyun_x.shape :  (180, 15) hyun_y.shpe :  (180,)
print('sam_x.shape : ',sam_x.shape,'sam_y.shape : ',sam_y.shape) # sam_x.shape :  (180, 15) sam_y.shape :  (180,)


# scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
hyun_x = scaler.fit_transform(hyun_x)
sam_x= scaler.transform(sam_x)
# hyun_x_train = scaler.fit_transform(hyun_x_train)
# hyun_x_test = scaler.transform(hyun_x_test)

timesteps = 15
sam_x = split_x(sam_x, timesteps)
hyun_x = split_x(hyun_x, timesteps)

print('hyun_x.shape : ',hyun_x.shape,'hyun_y.shpe : ',hyun_y.shape) # hyun_x.shape :  (165, 15, 15) hyun_y.shpe :  (180,)
print('sam_x.shape : ',sam_x.shape,'sam_y.shape : ',sam_y.shape) #sam_x.shape :  (165, 15, 15) sam_y.shape :  (180,)

sam_y = sam_y[timesteps:]
hyun_y = hyun_y[timesteps:]

sam_x_train, sam_x_test = train_test_split(
    sam_x,  train_size=0.80,shuffle=False
)
hyun_x_train, hyun_x_test = train_test_split(
    hyun_x,  train_size=0.80,shuffle=False
)
hyun_y_train, hyun_y_test  = train_test_split(
    hyun_y,train_size=0.80,shuffle=False
)
sam_y_train, sam_y_test  = train_test_split(
    sam_y,  train_size=0.80,shuffle=False
)
# sam_x_train = split_x(sam_x_train, timesteps)
# sam_x_test = split_x(sam_x_test, timesteps)
# hyun_x_train = split_x(hyun_x_train, timesteps)
# hyun_x_test = split_x(hyun_x_test, timesteps)
print(hyun_x_train.shape,hyun_x_test.shape) #(129, 15, 15) (21, 15, 15)
print(hyun_y_train.shape,hyun_y_test.shape) #(132,) (33,)
print(sam_x_train.shape,sam_x_test.shape) #(132, 15, 15) (33, 15, 15)
print(sam_y_train.shape,sam_y_test.shape) #(132,) (33,)


# 2. 모델 구성
#2-1. 모델 1
input1 = Input(shape=(timesteps,15))
lstm_BI01 = Bidirectional(LSTM(100, return_sequences=True,name='sam_x1'))(input1)
# lstm_BI01 = LSTM(100)(input1)
dense1 = Dense(80,activation='relu',name='hyun_x1')(lstm_BI01)
dense2 = Dense(20,activation='relu',name='hyun_x2')(dense1)
dense3 = Dense(30,activation='relu',name='hyun_x3')(dense2)
output1 = Dense(20,activation='relu',name='output_hyun')(dense3)

#2-2. 모델 2
input2 = Input(shape=(timesteps,15))
lstm_BI11 = Bidirectional(LSTM(100, return_sequences=True,name='sam_x1'))(input2)
dense12 = Dense(60,activation='relu',name='sam_x2')(lstm_BI11)
dense13 = Dense(40,activation='relu',name='sam_x3')(dense12)
dense14 = Dense(20,activation='relu',name='sam_x4')(dense13)
output2 = Dense(10,activation='relu',name='output_sam')(dense14)

# 2-3. 머지
from tensorflow.keras.layers import concatenate, Concatenate
merge1 = concatenate([output1,output2],name='mg1') # 두 개 이상이라서 리스트 
merge2 = Dense(20,activation='relu',name='mg2')(merge1)
merge3 = Dense(8,activation='relu',name='mg3')(merge2)
hidden_output = Dense(1,name='last')(merge3)

#2-4. 분기 모델 1
bungi1= Dense(10,activation='relu',name='result01')(hidden_output)
bungi2= Dense(10,activation='relu',name='result02')(bungi1)
last_output01= Dense(1,activation='relu',name='last_output01')(bungi2)

#2-4. 분기 모델 1
bungi1= Dense(10,activation='relu',name='result11')(hidden_output)
bungi2= Dense(10,activation='relu',name='result12')(bungi1)
last_output02= Dense(1,activation='relu',name='last_output02')(bungi2)
model = Model(inputs=[input1,input2], outputs=[last_output01,last_output02])
model.summary()

# 3. 컴파일 훈련 
model.compile(loss='mse',optimizer='adam')
import time
start = time.time()
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(
    monitor='loss',
    patience=50,
    mode='min',
    verbose=1,
    restore_best_weights=True
)
mcp=ModelCheckpoint(monitor='loss',mode='min',verbose=1, save_best_only=True,
                    filepath='./_save/samsung/keras53_samsung2_kjy_save.hdf5'
                    )
model.fit(
    [sam_x_train,hyun_x_train],
    [sam_y_train,hyun_y_train],
    epochs=500,batch_size=2000,
    validation_split=0.2,
    callbacks=[mcp]
)
end = time.time()


# 4. 평가 예측 

loss=model.evaluate([ sam_x_test , hyun_x_test],[sam_y_test , hyun_y_test])
print('loss : ', loss )
sam_x_test_pred =np.array(sam_x_test[-timesteps : ])
hyun_x_test_pred =np.array(hyun_x_test[-timesteps : ])
result = model.predict([sam_x_test_pred , hyun_x_test_pred])
result = np.array(result)
result = result.reshape(2,timesteps,15)
print('result : ',result)
print('result : ',result.shape) #(2, 15, 15, 1)
print('time : ',round((end-start),2))
