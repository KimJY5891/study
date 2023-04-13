import pandas as pd


df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']},
                    index=[0, 1, 2, 3])

#이런식으로 된 판다스에서 column A 과 column B 를 합치고 싶어 문자 데이터도 합치고 싶어 코드 부탁해

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                    'B': ['B4', 'B5', 'B6', 'B7'],
                    'C': ['C4', 'C5', 'C6', 'C7'],
                    'D': ['D4', 'D5', 'D6', 'D7']},
                   index=[4, 5, 6, 7])

df1['AB'] = df1['A'] + df1['B']
print(df1['AB'])


'''
0    A0B0
1    A1B1
2    A2B2
3    A3B3
'''
'''
df3 = pd.concat([df1, df2])
print(df1)
print(df2)
print(df3) 

'''
