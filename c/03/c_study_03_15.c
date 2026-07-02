#include <stdio.h>
int main() {
	// 대입 연산에 의한 수식의 값 : 저장된 값 
	float a = 1.5;
	int b;
	float c;
	c = b = a; // 오른쪽부터 왼쪽으로 연산 , b=a부터 먼저 연산
	// b는 int이기 때문에 소수값을 버림 그리고 다시 c로 저장하여 1.00000이 됨
	printf("b = %d. c = %f\n", b, c); // b = 1. c = 1.000000
}
