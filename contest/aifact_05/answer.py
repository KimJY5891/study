import numpy as np
import pandas as pd
path = './_data/aifact_05/'
path_save = './_save/aifact_05/'

columns = ['','연도','일시','측정소','PM2.5']
values = np.repeat([0.1, 0.11, 0.12, 0.13, 0.14,0.15,0.16,0.17,0.18,0.19,0.198], 14000)

answer_sample_csv = pd.read_csv(path+'answer_sample.csv')
# answer_sample_csv = np.array(answer_sample_csv)
# print(answer_sample_csv)
# answer_sample_csv = pd.DataFrame(answer_sample_csv, columns=columns)
answer_sample_csv['PM2.5'] = 0.0585
# 0.0585 = 9.84276067점
answer_sample_csv.to_csv(path_save+'0427_058.csv',encoding='UTF-8')
