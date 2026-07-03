#include <stdio.h>
int main() {
	// 조건 연산자
	int a, b; 
	printf("두 수를 입력하시오\n");
	scanf_s("%d %d", &a, &b);
	printf("최댓값?= %d\n", a>=b? a:b);
}
