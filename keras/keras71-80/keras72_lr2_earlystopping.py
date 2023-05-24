# 오류 수정요망
# [실습] earlystopping 구현 하기
# 1. 최소값을 넣을 변수를 하나 준비
# 2. 에포에 값과 최소값을 비교, 
# 최소값이 갱신되면 그 변수에 최소값을 넣어주고 카운트 초기화
# 3. 갱신이 안되면 카운트 변수 ++1
#  카운트 변수가 내가 원하는 얼리스타핑 갯수에 도달하면 for문을 stop

y_pred = 12
patience = 12

x= 10 
y = 10

w = 11
lr = 0.055
epochs = 183
count = 0 


for  i in range(epochs) :
    hypothesis = x * w 
    loss  = (hypothesis - y) ** 2 # mse
    print(f'{i+1}번째','Loss : ', round(loss,4),'\tPredict : ',round(hypothesis,4))
            
    up_predict = x *(w+lr)
    up_loss = (y - up_predict) **2 
    down_predict = x *(w+lr)
    down_loss = (y - up_predict) **2 
    
    if(up_loss >= down_loss) : 
        w = w - lr
    else : 
        w = w + lr

    if  loss_min > loss  :
        loss_min = loss
        count =+1
    if count == 10:
        break    
    print(f'{i+1}번째','Loss : ', round(loss,4),'\tPredict : ',round(hypothesis,4))
