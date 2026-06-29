//#define _CRT_SECURE_NO_WARNINGS 
// 아니면 vs는 scanf_s을 사용해야함
#include <stdio.h>
int main() {
	// 정수
	int a;
	short b;
	long long c;
	scanf_s("%d",&a);
	scanf_s("%hi",&b);
	scanf_s("%llx",&c);  // 여기까지 입력값받는 코드
	printf("%d %d %lld\n",a,b,c); // 10진수 unsigned long long으로 출력
	// 실수
	float f;
	double d;
	scanf_s("%f, %lf", &f, &d); // 입력받을때에는 변수 앞에 &
	printf("%f %e\n", f,d);
}
/*
입력 : 10, 0x20, 30
출력 : 10 32 48
입력 : 1.23, 45.9e-4
출력 : 1.230000 4.590000e-03
*/
