#pip install beautifulsoup4 requests
import requests
from bs4 import BeautifulSoup

# 크롤링할 웹 페이지의 URL
url = 'https://example.com'

# HTTP GET 요청 보내기
response = requests.get(url)

# 응답 내용을 파싱하기
soup = BeautifulSoup(response.content, 'html.parser')

# 웹 페이지의 제목 추출
title = soup.title.text
print('웹 페이지 제목:', title)

# 모든 링크 추출 및 출력
links = soup.find_all('a')
print('모든 링크:')
for link in links:
    print(link.get('href'))

# 원하는 정보를 더 추출하고 처리할 수 있습니다.
# 예를 들어 특정 클래스나 ID를 가진 요소를 찾아내거나, 다른 웹 페이지로 이동하여 정보를 수집할 수 있습니다.
