#include <stdio.h>
void main() {
	//2항연산자
	int x = 10, y = 3;
	printf("x = %d, y = %d\n",x,y );
	printf("x + y = %d\n",x + y);
	printf("x / y = %d\n",x / y);
	printf("x %% y = %d\n",x % y);
	printf("y %% x = %d\n",y % x);
}
/*
결과
x = 10, y = 3
x + y = 13
x / y = 3
x % y = 1
y % x = 3
*/
