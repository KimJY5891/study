#include <stdio.h>
int main() {
	// 콤마 연산의 사용 예 
	int i, j;
	for (i = 0, j = 4;i < 5;i++, j--) // 3개의 식 식안에 식이 있음
		printf("[%d,%d]\n", i, j);
}
