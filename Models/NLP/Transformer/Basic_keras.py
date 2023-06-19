#  이진 분류 

# 설정
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling1D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score


# Transformer 블록을 레이어로 구현
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# 임베딩 레이어 구현
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
# 1. 데이터

vocab_size = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)
print(len(x_train), "Training sequences")
print(len(x_test), "Validation sequences")

x_train = keras.utils.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.utils.pad_sequences(x_test, maxlen=maxlen)

print(f'y_train : ', y_train.shape) # y_train :  (25000,)
print(f'y_train : ', y_train) # y_train :  (25000,)
print(f'x_train : ', x_train.shape) # x_train :  (25000, 200)
print(f'x_train : {x_train}')

# 2. 모델 
# 변환기 계층을 사용하여 분류자 모델 만들기
# 트랜스포머 레이어는 입력 시퀀스의 각 시간 단계에 대해 하나의 벡터를 출력합니다.
# 여기서는 모든 시간 단계에서 평균을 취하고 그 위에 피드포워드 네트워크를 사용하여 텍스트를 분류합니다.

embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.1)(x)
x = Dense(20, activation="relu")(x)
x = Dropout(0.1)(x)
outputs = Dense(2, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# 3. 컴파일, 훈련 

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
es = EarlyStopping(monitor='val_acc',patience=32,mode='max',
               verbose=1,restore_best_weights=True)
history = model.fit(
    x_train, y_train, batch_size=128, epochs=1024, validation_split=0.2,
    callbacks=[es]
    )



# 4. 평가, 예측 

loss = model.evaluate(x_test,y_test)
print('loss : ',loss[0])
print('acc : ',loss[1])
# loss :  [0.3138861060142517, 0.8736400008201599]

# y_test = np.argmax(y_test,axis=1)
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis=1)
# y_pred= np.argmax(y_pred,axis=-1)
print(f'y_test : {y_test}')
print(f'y_prcd : {y_pred}')

acc = accuracy_score(y_test,y_pred)
print('acc : ',acc)
'''
y_test : [0 1 1 ... 0 0 0]
y_prcd : [[0.9281785  0.07182151]
 [0.00482307 0.9951769 ]
 [0.37385648 0.6261435 ]
 ...
 [0.9723512  0.02764882]
 [0.9588922  0.04110771]
 [0.6649947  0.3350053 ]]
'''

