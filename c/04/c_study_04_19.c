#include <stdio.h> 
void main() {
	int i = 0,n;
	int sum = 0;
	printf("n=?");
	scanf("%d", &n);
	do { //do while 루프
		sum = sum + i;
		i++;
	} while (i <= n);
	printf("i = %d \n", i);
	printf("i ~ %d까지의 합 = %d \n", i,n,sum);

}
