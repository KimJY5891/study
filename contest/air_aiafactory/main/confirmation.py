import pandas as pd
import numpy as np
save_path= './_save/aifactory/'
result_csv = pd.read_csv(
    save_path + 'submit_air_0418_0931.csv'
    ,index_col=0   
)
print(np.unique(result_csv['label'] )) # [0 1]
counts = result_csv['label'].value_counts()
'''
0    7020
1     369

결론 369~371의 1이 많다. 
'''
print(counts)
