#include <stdio.h>
void main() {
	int i, sum = 0; // 루프 변수 i는 정수형이어야함
	for (i = 1; i <= 10; ++i) { /* i값이 11이 되면 조건식이 거짓이 되어 루프를 빠져나옴*/
		sum = sum + i;
		printf("%d번째 실행 : sum = %d \n", i, sum);
	}
	printf("1부터 %d까지의 합 = %d \n", i - 1, sum);
}
