import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers import Dense,Input, Conv1D,Flatten
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_squared_error #mse에서 루트 씌우면 rmse로 할 수 있을지도?

# 1. 데이터
path='./_data/kaggle_jena/'
path_save='./_save/kaggle_jena/'

datasets=pd.read_csv(path+'jena_climate_2009_2016.csv',index_col=0)
print(datasets) #[420551 rows x 14 columns]

'''
시계 데이터는 보통 인덱스 
y =T (degC)
                  Date Time  p (mbar)  T (degC)  Tpot (K)  Tdew (degC)  rh (%)  VPmax (mbar)  VPact (mbar)  VPdef (mbar)  sh (g/kg)  H2OC (mmol/mol)  rho (g/m**3)  wv (m/s)  max. wv (m/s)  wd (deg)
0       01.01.2009 00:10:00    996.52     -8.02    265.40        -8.90   93.30          3.33          3.11          0.22       1.94             3.12       1307.75      1.03           1.75     152.3   
1       01.01.2009 00:20:00    996.57     -8.41    265.01        -9.28   93.40          3.23          3.02          0.21       1.89             3.03       1309.80      0.72           1.50     136.1   
2       01.01.2009 00:30:00    996.53     -8.51    264.91        -9.31   93.90          3.21          3.01          0.20       1.88             3.02       1310.24      0.19           0.63     171.6   
3       01.01.2009 00:40:00    996.51     -8.31    265.12        -9.07   94.20          3.26          3.07          0.19       1.92             3.08       1309.19      0.34           0.50     198.0   
'''
print(datasets.columns) #판다스
'''
Index(['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
       'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
       'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
       'wd (deg)'],
      dtype='object')
'''
print(datasets.info())
#non-null 결측치 음ㅍㅍ 
#Dtype : float64
print(datasets.describe()) # 평균 
#df.head - 위에서 5개만 보여준다. 

#판다스를 넘파이로 바꾸는 법 
print(datasets['T (degC)'].values) # 판다스를 넘파이로 바꾸는 것 : .values
print(datasets['T (degC)'].to_numpy()) # 판다스를 넘파이로 바꾸는 것 : .values

# import matplotlib.pyplot as plt
# plt.plot(datasets['T (degC)'].values)
# plt.show()
# 몇개를 자를지 
datasets, x_predict = train_test_split(
    datasets,
    train_size=0.9,random_state=8715,shuffle=False
)
print(datasets) #[420551 rows x 14 columns]
print('x_predict.shape:',x_predict.shape)  #(42056, 14)
print('x_predict:',x_predict)
print('datasets_test:',datasets)
timesteps = 10
def split_x(dataset, timesteps) :
    list =[]
    for i in range(len(dataset)-timesteps) :
        subset = dataset[i:(i+timesteps)]
        list.append(subset)
    return np.array(list)
x_predict = x_predict.drop(['T (degC)'],axis=1)
x_predict=split_x(x_predict,timesteps)
print('x_predict.shape:',x_predict.shape) 
x = datasets.drop(['T (degC)'],axis=1)
y = datasets['T (degC)']
x = split_x(x,timesteps)
# 내가 알아서 잘라야함 
# 선생님 말씀으로는 2번 

y=y[timesteps:]
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.67,random_state=8715,shuffle=False
)

#x=x.values.reshape(1328, 9, 1)
print('x_train:',x_train)
print('x_test:',x_test)
print('x_train.shape:',x_train.shape) 
print('x_test.shape:',x_test.shape) 
print('y_train.shape:',y_train.shape) 
print('y_test.shape:',y_test.shape) 
print('x_predict.shape:',x_predict.shape) 

# x_train.shape: (253587, 5, 13)
# x_test.shape: (124900, 5, 13)
# y_train.shape: (253591,)
# y_test.shape: (124904,)

#2.모델구성

Input01 = Input(shape=(10,13))
Conv1D01 = Conv1D(10,2,padding='same')(Input01)
Conv1D02 = Conv1D(10,2,padding='same')(Conv1D01)
Flatten01 = Flatten()(Conv1D02)
Dense01 = Dense(12)(Flatten01)
Dense02 = Dense(12)(Dense01)
Output01 = Dense(1)(Dense02)
model = Model(inputs=Input01,outputs=Output01)

#3. 컴파일, 훈련
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
                    filepath='./_save/MCP/keras48_Conv1D_01_boston_mcp.hdf5') #가중치 저장
model.fit(x_train,y_train,epochs=1000,batch_size=100,callbacks=[es,mcp])
end = time.time()

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss: ',loss)
y_predict = model.predict(x_test)

r2=r2_score(y_test,y_predict)
print('r2스코어 : ',r2)


# LSTM MSE 
# 7:2:1 = train : test :predict
# 시계열 데이터는 섞으면 안된다.
# 바로 10분뒤는 그렇게 어렵지 않ㄴ다. 
# 시간으로 맞추는거 함 
# RNN 그 다음 LSTM에서 받아들일때, ㅋ
# 온도가 다있읜 x에 넣고 해도 된다. 
# 없으면 좀 빼는 편 
# 결측치 있는거 모델 한번 돌려서 결측치 채워서 돌린다. 
# 결측치를 와이롤 잡고 결측치를 채우고 나서 다시 돌리기도 한다. 이럴 댄 다른 모델로 돌린다. 
