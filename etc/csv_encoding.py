import csv
import pandas as pd

path = 'C:\pytorch-transformer\data/'

train_csv = pd.read_csv(path + 'ko-en-translation_copy.csv')
test_csv = pd.read_csv(path + 'test_copy.csv')

train_csv.to_csv(path + 'ko-en-translation_utf.csv',encoding='utf-8')
test_csv.to_csv(path + 'test_utf.csv',encoding='utf-8')
