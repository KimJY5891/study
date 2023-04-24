import pandas as pd
df = pd.DataFrame({'A':[1,2,3,4,5],
                   'B':[10,20,30,40,50],
                   'C':[5,4,3,2,1]})
print(df) # 딕셔너리 형태와 리스트 형태로 합쳐서 만들수 있음 
# 대량의 데이터는 이렇게 만들지 않음 

correlations = df.corr()
print(correlations) # 너무 신뢰하면 안되고 단순하게 기울기만 보여주는 것. 1. 이라고해서 완전 같은게 아니다. 
# 성능이 구린 리니어모델로 돌린것 
