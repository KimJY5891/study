#include <stdio.h>
#pragma warning(disable:4996)

void main() {
	int a; 
	printf("정수입력(1~200) : ");
	scanf_s("%d", &a);
	if (a < 100)
		printf("입력한 정수(%d)가 100보다 작음. \n", a);
	else
		printf("입력한 정수(%d)가 100보다 큼. \n", a);
	printf("프로그램 종료 \n");
}
