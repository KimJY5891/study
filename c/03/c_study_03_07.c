#include <stdio.h>
void main() {
	char var = 'A';
	putchar(var); // 함수의 괄호 안에 문자형 변수가 사용됨
	putchar(var+1); // 수식이 사용됨
	putchar('\n'); // escape 문자가 사용됨
	putchar('K'); // 문자형 상수가 사용됨
	putchar('K'+2); // 수식이 사용됨
	putchar('\007'); // escape 문자가 사용됨
}
// 출력 : 
/*AB
KM*/
