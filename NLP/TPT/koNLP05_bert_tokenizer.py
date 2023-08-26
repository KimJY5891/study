
from transformers import *
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
def bert_tokenizer(sent, MAX_LEN):
      
    encoded_dict = tokenizer.encode_plus(
          text = sent,
          add_special_tokens = True, # 시작점에 CLS, 끝점에 SEP가 추가된다.
          max_length = MAX_LEN,
          pad_to_max_length = True,
          return_attention_mask = True
      )
      
    input_ed = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    token_type_id = encoded_dict['token_type_ids']
    return input_id, attention_mask, token_type_id
