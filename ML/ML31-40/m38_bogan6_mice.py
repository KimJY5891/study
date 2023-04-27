# MICE(Multiple Inmputation by Chained Equations)
import numpy as np
import pandas as pd
from impyute.imputation.cs import mice

data =pd.DataFrame([[2,np.nan,6,8,10],
                    [2,4,np.nan,8,np.nan],
                    [2,4,6,8,10],
                    [np.nan,4,np.nan,8,np.nan]]
                   ).transpose()
print(data) # (5, 4)
data.columns = ['x1','x2','x3','x4']
print(data) # (4, 5)
'''
     x1   x2    x3   x4
0   2.0  2.0   2.0  NaN
1   NaN  4.0   4.0  4.0
2   6.0  NaN   6.0  NaN
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN
'''

impyute_df = mice(data.values)
# AttributeError: 'DataFrame' object has no attribute 'as_matrix'
# as_matrix - mice에서는 판다스가 안된다. 
# 넘파이로 변경해서 사용 
# data.values / data.to_numpy() / np.array()
'''
[[ 2.          2.          2.          1.91508158]
 [ 4.02423084  4.          4.          4.        ]
 [ 6.          5.89555505  6.          5.93587119]
 [ 8.          8.          8.          8.        ]
 [10.          9.7911101  10.          9.9566608 ]]
얘도 선형방식으로 찾는다. 
선형방식과 컬럼과의 상관관계를 확인하는 알고리즘 사용
'''
print(impyute_df)
