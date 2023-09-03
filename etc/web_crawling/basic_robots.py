import requests
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup

# 크롤러의 User-Agent 설정
user_agent = 'MyCrawler'
url = 'https://example.com'
robots_txt_url = f'{url}/robots.txt'

# robots.txt 파일을 파싱하여 지침을 읽는 함수
def check_robots_txt(url, user_agent):
    rp = RobotFileParser()
    rp.set_url(url)
    rp.read()
    return rp.can_fetch(user_agent, url)

# robots.txt를 확인하여 크롤링할 수 있는지 검사
if check_robots_txt(robots_txt_url, user_agent):
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 원하는 정보를 추출하고 처리하는 코드 작성
        # 예를 들어, 제목과 링크 추출
        title = soup.title.text
        print('웹 페이지 제목:', title)
        
        links = soup.find_all('a')
        print('모든 링크:')
        for link in links:
            print(link.get('href'))
    else:
        print('웹 페이지를 가져오는데 실패했습니다.')
else:
    print('robots.txt에 따라 해당 웹 페이지를 크롤링할 수 없습니다.')
