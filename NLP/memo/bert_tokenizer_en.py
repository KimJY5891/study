'''
출처 : https://wikidocs.net/166796
토크나이저를 로드하고 저장하는 것은 모델의 경우처럼 간단합니다. 실제로, 모델을 로드하고 저장할 때와 같이,
from_pretrained() 및 save_pretrained() 메서드(method)를 그대로 사용합니다.
이들 메서드(method)들은 토크나이저(모델의 아키텍처와 약간 비슷함)와
어휘집(vocabulary, 모델의 가중치(weights)와 비슷함)에서 사용하는 알고리즘을 로드하거나 저장합니다.
'''

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
print(tokenizer("Using a Transformer network is simple"))
#{'input_ids': [101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102],
# 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0],
# 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

print(tokenizer("Using a Transformer network is simple"))
# {'input_ids': [101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102], 
# 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0], 
# 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}


tokenizer.save_pretrained("saving_folder")

