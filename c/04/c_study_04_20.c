#include <stdio.h>
#pragma warning(disable:4496)
void main() {
	int num, sum = 0;
	while (1) { // 무한 while 루프 1은 참이니까~
		printf("num(끝:0)....?");
		scanf("%d", &num);
		if (num == 0) // num의 값이 0이면 while 루프 종료
			break; // 루프 바깥으로 빠져나감
		sum = sum + num;
	}
	printf("sum = %d", sum);
}
