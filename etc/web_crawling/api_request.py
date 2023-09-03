import requests

url = 'https://www.example.com'
response = requests.get(url)

if response.status_code == 200:
    print('Success!')
    print(response.text)  # 웹 페이지 내용 출력
else:
    print('Failed to retrieve the webpage')

'''
import requests는 파이썬 프로그래밍에서 사용되는 명령어로, requests 라이브러리를 현재 작업 중인 프로그램에 가져오는 역할을 합니다.
이 라이브러리는 웹 요청을 보내고 받는 기능을 제공하여 웹 서버와의 상호 작용을 용이하게 만들어줍니다.

requests 라이브러리를 사용하면 다양한 HTTP 요청 메서드(GET, POST, PUT, DELETE 등)를 사용하여 웹 서버로 데이터를 보낼 수 있고,
서버로부터 응답을 받아올 수 있습니다. 이를 통해 웹 페이지의 내용을 가져오거나,
API 엔드포인트에 데이터를 전송하거나, 파일을 다운로드하거나 업로드하는 등의 작업을 수행할 수 있습니다.

위의 코드에서 requests.get() 함수는 지정된 URL로 GET 요청을 보내고,
서버로부터의 응답을 response 객체로 받아옵니다. 이후에는 response 객체의 status_code 속성을 사용하여 응답 상태 코드를 확인하고,
성공적인 경우에는 웹 페이지 내용을 response.text를 통해 출력합니다.

요약하자면, import requests는 파이썬 코드에서 HTTP 요청을 보내고 받을 수 있게 해주는 requests 라이브러리를 가져오는 역할을 합니다.
이를 통해 웹 서버와의 효율적인 상호 작용이 가능해집니다.






'''
