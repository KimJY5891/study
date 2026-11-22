#include <stdio.h>
void main() {
	int i = 1, sum = 0; // while문 조기식(i=1)
	while (i <= 10) {//i값이 11이 되면 조건식이 거짓이 되어 루프를 빠져나옴\
		sum = i + sum;
		printf("%d번째 실행 : %d\n", i, sum);
		++i; //while문증감식(i++)
	}
	printf("1부터 %d까지의 합 = % \n", i - 1, sum);
}
