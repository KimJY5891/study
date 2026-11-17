#include <stdio.h>
//if문의 조건에서 참과 거짓의 판별은 어떻게 하는가
// c언어에서는 단지 0만 거짓으로 인식하고, 0이 아닌 다른 모든 값은 참으로 인식
void main() {
	int a = 50;
	printf("%d\n", a); // 50
	printf("%d\n", !a); // 0
	if (a)
		printf("TRUE");
	if (!a)
		printf("FALSE");
}
