#문자열 개수 세기
a = "hobby"
#count함수 : a에 b가 몇개 있는지 세줌
print(a.count('b'))#2
#find 함수 : 위치를 알려줌, 가장 먼저 나오는 b를 찾아준다. 
#표현은 인덱스로 해준다. 
print(a.find('b'))#2
print(a.find('y'))#4
print(a.find('h'))#0
print(a.find('x'))#-1
print(a.find('ㅅ'))#-1
#인덱스 함수
print(a.index('b'))
#join 함수
a=",".join("abcd")#a,b,c,d
print(a)
b=",".join(["a","b","c"]) 
print(b)#a,b,c
#대문자<->소문자
a="hi"
print(a.upper())#HI
print(a.lower())#hi
#strip 함수 : 공백 없애줌
a="       hㅑ       "

print(a.strip())#hㅑ
#제거 후 대체
a = """life is too short"""
print(a.replace("life","your leg"))#your leg is too short
