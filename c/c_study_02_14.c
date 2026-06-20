#include <stdio.h>
#define DEBUG_MODE 1
int main() {
	int a = 10, b = 20;
#if DEBUG_MODE
	printf("평균을 구할 값 : %d, %d\n", a, b);
#endif
	printf("평균= %f\n", (a + b) / 2.0);

} // ?됯퇏??援ы븷媛?: 10, 20
// ? 됯퇏 = 15.000000
