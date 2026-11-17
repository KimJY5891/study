#include <stdio.h>
#pragma warning(disable:4996)
#include <stdio.h>

void main() {
	int year, month, day;
	printf("말일 검색하고 싶은 연도와 월을 입력 : ");
	scanf("%d %d", &year, &month);/* 연도와 월을 순서대로 입력받아 변수 year, month에 저장*/
	if (month >= 1 && month <= 12) {// 입력된 연도가 1월부터 12월 사이인지 확인
		if (month == 2) {//2월 경우 윤년 여부 계산
			if (year % 4 == 0 && year % 100 != 0 || year % 400 == 0) {
				day = 29;
				printf("%d년 %d월은 %d일까지이고 윤년입니다. \n",year,month,day);
			}
			else {
				day = 28;
				printf("%d년 %d월은 %d일까지 입니다. \n", year, month, day);
			}
			
		}
		else { // 2월을 제외한 나머지 달의 말일 출력
			if (month == 1 || month == 3 || month == 5 || month == 7 || month == 8 || month == 10 || month == 12) {
				day = 31;
				printf("%d년 %d월은 %d일까지 입니다. \n", year, month, day);
			}
			else {
				day = 30;
				printf("%d년 %d월은 %d일까지 입니다. \n", year, month, day);
			}
		}
	}
	else 
		printf("잘못된 월을 입력 \n");
}
