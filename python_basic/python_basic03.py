b='12123425125'
print(b[::-1]) # 52152432121 역순으로 출력

#포메팅
"""
문자열 포맷 코드
%s : 문자열
(s는 숫자로 입력해도 문자열로 바뀌어서 입력된다.)
%c : 문자 1개
%d : 정수
%f : 부동 소수
%o : 8진수
%x : 16진수
%% : Literal%(문자‘%’ 자체)
"""

a="i eat %d apples."%3
print(a) #i eat 3 apples.

number=10
day = "thr"
a ="i ate %d apple, so i was sick %s days." %(number,day)
print(a) #i ate 10 apple, so i was sick three days.
