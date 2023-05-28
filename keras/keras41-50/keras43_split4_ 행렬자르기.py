import numpy as np

# 1. 데이터 
datasets= np.array(range(1,41)).reshape(10,4)
print(datasets) #10행 4열

x_data =datasets[:,:-1]
y_data =datasets[:,-1]
#x:(행,열):3
print('x_data:',x_data) #10행 4열
print(y_data) #10행
print(x_data.shape) #10행
print(y_data.shape) #10행
timesteps = 3

##### x만들기 #####
def split_x(dataset, timesteps) :
    aaa =[]
    for i in range(len(dataset)-timesteps) :
        subset = dataset[i:(i+timesteps)]
        aaa.append(subset)
    return np.array(aaa)
x_data = split_x(x_data, timesteps)
print('x_data:',x_data)
'''
x_data: [[[ 1  2  3]
  [ 5  6  7]
  [ 9 10 11]]

 [[ 5  6  7]
  [ 9 10 11]
  [13 14 15]]

 [[ 9 10 11]
  [13 14 15]
  [17 18 19]]

 [[13 14 15]
  [17 18 19]
  [21 22 23]]

 [[17 18 19]
  [21 22 23]
  [25 26 27]]

 [[21 22 23]
  [25 26 27]
  [29 30 31]]

 [[25 26 27]
  [29 30 31]
  [33 34 35]]]
'''
##### y만들기 #####
# 타임스텝스만큼 빼기
y_data=y_data[timesteps:]
print(y_data) # [16 20 24 28 32 36 40]
