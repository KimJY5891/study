#include <stdio.h>
#define C_AREA(x) (3.141592 * (x) * (x))

int main() {
	double r = 10.0;
	printf("%f\n", C_AREA(r));
} // 314.159200
