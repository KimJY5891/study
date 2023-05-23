import numpy as np

f = lambda x : x**2 -4*x +6
# def f(x) : 
#     return x**2 - 4*x +6

# gradient = f(x)미분
gradient =  lambda x : 2*x -4

x = 10.0 # 초기값 
epochs  =20
learning_rate = 0.25
x_list =[]
fx_list = []
for i in range(epochs) : 
    x = x - learning_rate*gradient(x)
    # print(i+1,'번째','\t',x,'\t',f(x))
    print("{:02d}\t {:6.5f}\t {:6.5f}\t".format(i+1,x,f(x)))
    x_list.append(x)
    fx_list.append(f(x))


import matplotlib.pyplot as plt

plt.plot(x_list,fx_list,'k-') 
plt.plot(x_list[9],fx_list[9],'sk',color='red') 
plt.plot(x_list[19],fx_list[19],'sk',color='red') 
plt.grid()
plt.xlabel('x')
plt.ylabel('fx')
plt.show()
