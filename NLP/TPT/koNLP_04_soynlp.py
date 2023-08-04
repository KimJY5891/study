'''
요약 : 
비지도 학습 soynlp 
어느정도 규모가 있으면서 동질적인 문서 집합

출처 : https://github.com/lovit/soynlp , https://wikidocs.net/92961
한국어 분석을 위한 pure python code 입니다.
학습데이터를 이용하지 않으면서 데이터에 존재하는 단어를 찾거나, 
문장을 단어열로 분해, 혹은 품사 판별을 할 수 있는 비지도학습 접근법을 지향합니다.
비지도학습 기반 접근법들은 통계적 패턴을 이용하여 단어를 추출하기 때문에 
하나의 문장 혹은 문서에서 보다는 어느 정도 규모가 있는 동일한 집단의 문서 (homogeneous documents) 에서 잘 작동합니다. 
영화 댓글들이나 하루의 뉴스 기사처럼 같은 단어를 이용하는 집합의 문서만 모아서 Extractors 를 학습하시면 좋습니다. 
이질적인 집단의 문서들은 하나로 모아 학습하면 단어가 잘 추출되지 않습니다.
soynlp는 품사 태깅, 단어 토큰화 등을 지원하는 단어 토크나이저입니다. 
비지도 학습으로 단어 토큰화를 한다는 특징을 갖고 있으며, 데이터에 자주 등장하는 단어들을 단어로 분석합니다.
soynlp 단어 토크나이저는 내부적으로 단어 점수 표로 동작합니다. 
이 점수는 응집 확률(cohesion probability)과 브랜칭 엔트로피(branching entropy)를 활용합니다.
'''
# 기존 형태소 분석기 
from konlpy.tag import Okt

tokenizer = Okt()
print(tokenizer.morphs('에이비식스 이대휘 1월 최애돌 기부 요정'))
# morphs : 형태소 추출
# pos : 품사태깅 (Part of speech tagging)
# nouns : 명사 추출
# ['에이', '비식스', '이대', '휘', '1월', '최애', '돌', '기부', '요정']
# 기존 형태소 분석기의 문제점 : 에이비식스와 이대휘는 이름인데도 형태소로 분리했다.

import urllib.request
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor
path = 'c:/study_koNLP/_data/'

with open(
    path+'2016-10-20.txt', # 경로
    'r',# r의 경우  read(읽기), w의 경우 write(작성), a의 경우 add(추가)       
    encoding='utf-8'
) as f :
    line_file = f.readlines() # f에서 모든 줄을 읽어들이며 리스트로 반화하는 메소드
# line_git = urllib.request.urlretrieve("https://raw.githubusercontent.com/lovit/soynlp/master/tutorials/2016-10-20.txt", filename="2016-10-20.txt")
# print(len(line_git))
print(line_file)
print(len(line_file)) # 29719
# print(line_git)


# 훈련 데이터를 다수의 문서로 분리
corpus = DoublespaceLineCorpus(path +"2016-10-20.txt")
print(len(corpus)) # 1801

# 3개만 보기
i = 0
for document in corpus : 
    if len(document) > 0:
        print('순서',i)
        print(document)
        i = i+1
    if i ==3 :
        break
    
# soynlp는 학습 기반의 단어 토크나이저이므로 기존의 KoNLPy에서 제공하는 형태소 분석기들과 달리 학습 과정을 거쳐야합니다.
word_extractor =WordExtractor()
word_extractor.train(corpus)
word_score_table = word_extractor.extract()
#  extractor : 추출기, extract : 추출하다.
'''
학습 완료시 나옴
training was done. used memory 0.998 Gb
all cohesion probabilities was computed. # words = 220556
all branching entropies was computed # words = 357480
all accessor variety was computed # words = 357480
'''
