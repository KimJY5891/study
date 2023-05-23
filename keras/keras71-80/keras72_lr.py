# X와 Y는 바꾸지 말고 다른거 바꾸면서 좋은 결과값이 나오게 맞추기 
x= 10 
y = 10
w = 11
lr = 0.055
epochs = 183
# 웨이트의 초기값도 중요하기 대문에 이니셜라이져? 라는 걸로 초기값을 잡는것을 한다. 
# 183번째 Loss :  0.01    Predict :  9.9

for  i in range(epochs) :
    hypothesis = x * w 
    loss  = (hypothesis - y) **2 # mse
    print(f'{i+1}번째','Loss : ', round(loss,4),'\tPredict : ',round(hypothesis,4))
    
    up_predict = x *(w+lr)
    up_loss = (y - up_predict) **2 
    down_predict = x *(w+lr)
    down_loss = (y - up_predict) **2 
    if(up_loss >= down_loss) : 
        w = w - lr
    else : 
        w = w + lr

