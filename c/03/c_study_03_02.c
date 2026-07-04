#include <stdio.h>
void main() {
	//출력양식변환기호 활용 예 - 정수형
	printf("%d\n",0x10);
	printf("%x\n",125 );
	printf("%X\n",125);
	printf("%o\n",125);
	// 출력양식 변환기호 활용 예 - 문자 
	printf("%c\n", 'A');
	printf("%c\n", 0x42);
	printf("%s\n", "KJY");
	printf("%s\n", "ab\0cd");

}
/* 결과 :
16 (10진수로 변환결과)
7d (16진수로 변환결과)
7D (16진수로 변환결과)
175 (8진수로 변환결과)
A
B
KJY
ab
*/
