
#문자열 나누기
# 문자열 자료형이 있으면 띄어쓰기를 기준으로 리스트를 만들어준다.
my_string = """life is too short"""
my_list = my_string.split()
print(my_list)
#split(":") 이런식으로 작성하면 :을 기준으로 리스트를 만들어준다.
b = "ㅁ:ㄴ:ㅇ:ㄹ"
print(b.split(":"))
#리스트
#변수를 하나로 묶는 역할을 한다.
a =["b","c","d","e","f"]
print(a)
print(a[0])
print(a[1])
print(a[4])
#  리스트[인덱스값]
#리스트 종류
#빈값
a=[]
#숫자
b=[1,2,3,4,5]
#문자
c =["b","c","d","e","f"]
#숫자+문자   
d = [1,2,3,4,"a","b","c"]
print(d)
#리스트안에 리스트 넣기 가능
e = ["01","02",[1,2,3,"a","b"]]
print(e)
