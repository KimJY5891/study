# 공부 자료 출처 : https://wikidocs.net/92961
# 한국어 전처리 패키지(Text Preprocessing Tools for Korean Text)
# pip install git+https://github.com/haven-jeon/PyKoSpacing.git
# PyKoSpacing : 한국어 자연어 처리를 위한 모듈로, 띄어쓰기 교정 기능을 제공
import numpy as np
import pandas as pd


sent = '김철수는 극중 두 인격의 사나이 이광수 역을 맡았다. 철수는 한국 유일의 태권도 전승자를 가리는 결전의 날을 앞두고 10년간 함께 훈련한 사형인 유연재(김광수 분)를 찾으러 속세로 내려온 인물이다.'


new_sent = sent.replace(" ",'') # 띄어쓰기가 없는 문자 임의로 만들기
print(new_sent)
#김철수는극중두인격의사나이이광수역을맡았다.철수는한국유일의태권도전승자를가리는결전의날을앞두
#고10년간함께훈련한사형인유연재(김광수분)를찾으러속세로내려온인물이다.

from pykospacing import Spacing
spacing = Spacing()
kospacing_sent = spacing(new_sent)
print(kospacing_sent)
'''
김철수는 극중 두 인격의 사나이 이광수 역을 맡았다. 철수는 한국 유일의 태권도 전승자를 가리는 결 
전의 날을 앞두고 10년간 함께 훈련한 사형인 유연재(김광수 분)를 찾으러 속세로 내려온 인물이다.
'''
