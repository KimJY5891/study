#include <stdio.h>
// 단순 if문 예시01
void main() {
	int a;
	printf("정수입력 :(1 ~ 200 ) : ");
	scanf_s("%d", &a); // 정수를 입력받아 변수 a에 저장
	if (a < 100) // 변수 a의 값이 100보다 크면 printf문 실행 안함
		printf("입력한 변수가 100보다 작음. \n");
	printf("a = %d", a);

}
