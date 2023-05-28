import numpy as np

dataset = np.array(range(1,11))

timesteps = 5
def split_x(dataset, timesteps) :
    aaa =[]
    for i in range(len(dataset)-timesteps +1) :
        # 반복할 거야 range(len(dataset)-timesteps +1)만큼
        # 변수 i 반복할 때마다 카운트가 올라감
        # for i in a = 변수 i가 a까지
        subset = dataset[i:(i+timesteps)]
        aaa.append(subset)
    return np.array(aaa)
'''
1. (테이터셋의전체길이 - 데이터를자르고싶은단위수 + 1)까지 0부터 반복을 시작
2. aaa리스트는 비어있는 채로 존재하고
3. 매개변수이상 (매개변수 + 데이터를자록싶은단위수)미만으로 자를것이야
4. aaa변수에 리스트로 추가해라
5. (테이터셋의길이 - 데이터를자르고싶은단위수 + 1)이 아니라면 계속 반복해라
6. 다 반복하면 주어진 변수를 numpy 배열로 변환해줘
자르고 싶은 만큼 줄어든다.
'''

bbb= split_x(dataset,timesteps)
print(bbb)
print(bbb.shape) #(6,5)
# 5개씩 잘랐다.
'''X          |  Y
[[ 1  2  3  4 |  5]
 [ 2  3  4  5 |  6]
 [ 3  4  5  6 |  7]
 [ 4  5  6  7 |  8]
 [ 5  6  7  8 |  9]
 [ 6  7  8  9 | 10]]
(6, 5)
'''
#X=bbb[:,:4]
X=bbb[:,:-1]
y=bbb[:,-1]
print(X)
print(y)
