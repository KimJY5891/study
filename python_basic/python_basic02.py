# 문자열의 합
a="python" 
b="is fun" 
print(a+b) #pythonis fun

# 문자열의 곱
print(a*5) #pythonpythonpythonpythonpython

# 인덱싱 
# 맨 앞의 인덱스는 0이다. 
a="python"
print(a[0]) #p
print(a[1]) #y
print(a[-1]) #n
print(a[-2]) #o
# 마이너스 인덱스는 뒤에서 꺼꾸로 세기 시작한다.

# 슬라이싱
a= "Life is too short"
print(a[0:4])#Life
# 0이상 8미만 2칸 간격
print(a[0:8:2])#Lf s
# 처음부터 4미만 1칸 간격
print(a[:4]) #Life
# 4부터 끝까지 1칸간격
print(a[4:]) # is too short
# 처음부터 끝까지 2칸간격
print(a[::2])# is too short
# -2의 경우 두 칸씩 뒤로 간다. 
print(a[::-2]) #tosots fL
print(a[::]) #Life is too short
print(a[::-1]) #trohs oot si efiL
