
'''
파라미터에 대한 자동화 툴
텐서와 사이킷런이 서로 달라서 랩핑하는 형식으로 해줘야한다. 

'''
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense,Conv2D,Flatten, MaxPool2D,Input, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
filpath = './_save/MCP/keras66_1_hyperParameter.hdf5'

# 1. 데이터 
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28*28).astype('float32')/255.
# x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2]).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0],-1).astype('float32')/255.
# print(x_test.shape)

# 2. 모델
def bulid_model(drop=0.5,optimizer='adam', activation='relu', node1= 64, node2 = 64, node3=64, lr=0.001,
                num_hidden_layer=3,patience = 30,monitor='val_loss') :
    inputs = Input(shape = (28*28),name='Input')
    x = Dense(512,activation=activation, name='hidden1')(inputs) 
    x = Dropout(drop)(x)
    for i in range(num_hidden_layer):
        x = Dense(node1, activation=activation)(x)
        x = Dropout(drop)(x)
    outputs = Dense(10,activation='softmax',name='outputs')(x)
    
    model = Model(inputs= inputs,outputs=outputs)
    model.compile(optimizer = optimizer, metrics=['acc'], loss = 'sparse_categorical_crossentropy', )   
    return model

def create_hyperparameter() : 
    batchs = [100,200,300,400,500]
    optimizers = ['adam','rmsprop','Adadelta']
    dropouts = [0.2,0.3,0.4,0.5]
    activation = ['relu','elu','selu','linear']
    node1 = [16, 32,64,128,256]
    node2 = [16, 32,64,128,256]
    node3 = [16, 32,64,128,256]
    lr = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    num_hidden_layers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    return {
        'batch_size': batchs,
        'optimizer' : optimizers,
        'drop' : dropouts,
        'activation':activation,
        'node1' : node1,
        'node2' : node2,
        'node3' : node3,
        'lr':lr, # 성능향상에 기여가 큼
        'num_hidden_layer':num_hidden_layers
        
        }

hyperparameters = create_hyperparameter()
# print(hyperparameters)

model01 = bulid_model() # 처음에 넣었던 방식  

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
keras_model =KerasClassifier(build_fn=bulid_model,verbose=1,epochs=3)
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

# model = GridSearchCV(
#     # model01,
#     keras_model,
#     hyperparameters,
#     cv=2
# )
model = RandomizedSearchCV(
    # model01,
    keras_model,
    hyperparameters,
    n_iter=1,
    cv=2,
    verbose=1
)# 텐서플로우와 사이킷런에서 만든 것이 공유가 안될 수도? 
import time
start = time.time()
mcp = ModelCheckpoint(
        monitor='val_loss',mode='auto',
        verbose=1, save_best_only=True, filepath=filpath
)
es = EarlyStopping(monitor='val_loss',patience=45,mode='auto',
               verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,verbose=1,epochs=128,callbacks = [es,mcp],validation_split = 0.2)



end = time.time()

print('걸린시간 : ',end-start)
print('model.best_estimator_ : ',model.best_estimator_)
print('model.best_params_ : ',model.best_params_)
print('model.best_score_ : ', model.best_score_)  # train data
print('model : ', model.score(x_test,y_test)) #  test data

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print('acc : ',accuracy_score(y_test,y_pred)) #  = model.score(x_test,y_test)

