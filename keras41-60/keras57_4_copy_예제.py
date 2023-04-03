import numpy as np

aaa = np.array([1,2,3])
bbb = aaa

bbb[0] = 4
print(bbb) #[4 2 3]
print(aaa) #[4 2 3]
# aaa 의 주소 값이 바뀌었다. 
# 카피를 했을 경우 주소 값이 공유가 된다. 
# 이런식으로 작성하면 메모리가 공유되어서 바뀐다.
# 그래서 이렇게 안되게 하려면 
print('====================================')
ccc = aaa.copy()
# 그러면 새로운 메모리 구조가 생성되는것
ccc[1] = 7
print(bbb) # [4 7 3]
print(aaa) # [4 2 3]
