from sklearn.datasets import fetch_california_housing, load_diabetes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score

# 1. 데이터
datasets = load_diabetes()
df = pd.DataFrame(datasets.data,columns=datasets.feature_names)
df['target'] = datasets.target
print(df)
# 그래프 확인

df.info() 
print(df.describe())


# x.plot.box()


#그래프가 한쪽으로 치우쳐있을 경우 
# df['target'].hist(bins=50 ) 


'''
                age           sex           bmi            bp            s1            s2            s3            s4            s5            s6      target
count  4.420000e+02  4.420000e+02  4.420000e+02  4.420000e+02  4.420000e+02  4.420000e+02  4.420000e+02  4.420000e+02  4.420000e+02  4.420000e+02  442.000000
mean  -3.634285e-16  1.308343e-16 -8.045349e-16  1.281655e-16 -8.835316e-17  1.327024e-16 -4.574646e-16  3.777301e-16 -3.830854e-16 -3.412882e-16  152.133484
std    4.761905e-02  4.761905e-02  4.761905e-02  4.761905e-02  4.761905e-02  4.761905e-02  4.761905e-02  4.761905e-02  4.761905e-02  4.761905e-02   77.093005
min   -1.072256e-01 -4.464164e-02 -9.027530e-02 -1.123996e-01 -1.267807e-01 -1.156131e-01 -1.023071e-01 -7.639450e-02 -1.260974e-01 -1.377672e-01   25.000000
25%   -3.729927e-02 -4.464164e-02 -3.422907e-02 -3.665645e-02 -3.424784e-02 -3.035840e-02 -3.511716e-02 -3.949338e-02 -3.324879e-02 -3.317903e-02   87.000000
50%    5.383060e-03 -4.464164e-02 -7.283766e-03 -5.670611e-03 -4.320866e-03 -3.819065e-03 -6.584468e-03 -2.592262e-03 -1.947634e-03 -1.077698e-03  140.500000
75%    3.807591e-02  5.068012e-02  3.124802e-02  3.564384e-02  2.835801e-02  2.984439e-02  2.931150e-02  3.430886e-02  3.243323e-02  2.791705e-02  211.500000
max    1.107267e-01  5.068012e-02  1.705552e-01  1.320442e-01  1.539137e-01  1.987880e-01  1.811791e-01  1.852344e-01  1.335990e-01  1.356118e-01  346.000000
'''
df.hist(bins=50)
plt.show()

y = df['target']
x = df.drop(['target'],axis=1)

# x['bmi'] = np.log1p(x['bmi'])
# x['bp'] = np.log1p(x['bp']) 
# x['s2'] = np.log1p(x['s2']) 
# x['s3'] = np.log1p(x['s3']) 
# x['s4'] = np.log1p(x['s4']) 
# x['s5'] = np.log1p(x['s5'])
# y = np.log1p(y)

x_train , x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, train_size=0.8, random_state=337
)

y_train_log = np.log1p(y_train)
y_test_log =  np.log1p(y_test)

# 2. 모델 
model = RandomForestRegressor(random_state=337)


# 3. 컴파일, 훈련 
model.fit(x_train,y_train_log)

# 4. 평가, 예측 
score = model.score(x_test,y_test)
print('로그 -> 지수r2 : ',r2_score(y_test,np.expm1(model.predict(x_test))))

#  0.3361676904683043
#    로그 -> 지수r2 : 0.38027974872160075
