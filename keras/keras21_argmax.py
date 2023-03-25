import numpy as np

a = np.array ([[1,2,3],
               [6,4,5],
               [15,9,2],
               [3,2,1],
               [2,3,1]]) #(5,3)
print (a.shape) #(5, 3)
print (np.argmax(a)) #(7 7 번째 자리가 가장 높다. 
print (np.argmax(a,axis=0)) # [221] axis = 0은 행이다. 열기준(세로로) 으로에서 비교하여 가장 높은 행의 인덱스로 알려준다.  
print (np.argmax(a,axis=1)) # [2 0 0 0 1] axis = 1은 이다. 행안에서 열끼리 비교 행에서 가장 높은 열 
print (np.argmax(a,axis=-1)) # 가장 마지막 축, 이건 2차원이니가 가장 마지막 축은  1, # 그래서 -1을 쓰면 이 데이터는 1과 동일 
 
