//다중 if ~ else 문의 사용
#include <stdio.h>
#pragma warning(disable:4996)
#include <stdio.h>

void main() {
	int a;
	printf("정수를 입력하세요 : ");
	scanf_s("%d", &a); // 정수를 입력 받아 변수 a에 저장
	if (a >= 0)
		if (a == 0) // 첫 번째와 두번째 조건을 모두 만족할 경우 출력
			printf("입력된 값은 0");
		else // 첫 번재ㅔ 조건을 만족하고 두 번째 조건을 만족하지 않을 경우 출력
			printf("입력된 값음 양수");
	else
		printf("입력된 값은 음수");
}
