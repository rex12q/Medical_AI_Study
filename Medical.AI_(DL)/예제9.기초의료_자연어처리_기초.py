import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
#특징_추출: 기계가 이해할 수 있도록 정형화된 데이터를 학습하여 변환해주는 역할 (그래서 이번 주제에 맞춰서 .text를 씀)
#CountVectorizer: 단어가 몇 번 등장했는지 빈도수를 세어서 숫자로 가득 찬 DF(matrix) 생성 

#1. 임의의 병원 진료 기록
h_text=[
    'Patient has a high fever and cough.', # 열, 기침 둘 다
    'Patient has no fever but has a cough.', #열은 없지만, 기침
    'Doctor prescribed fever medicine.' #해열제 처방
]

#모듈 호출
vectorizer=CountVectorizer() #Count language
#학습 후 변환
X=vectorizer.fit_transform(h_text)
print('사전 출력 결과')
print(vectorizer.get_feature_names_out())
#get_feature_names_out: 각각 어떤 단어(이름)에 매칭되는지 순서대로 글자 이름표만 뽑아내 주는 내장 함수
print('-'*50)

#기계 시점) 데이터 형태(배열) 출력
print('기계 시점 출력 결과')
df=pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names_out())
#toarray: 압축된 형태의 희소 행렬을 사용자가 눈으로 보고 계산할 수 있는 2차원 숫자 배열(numpy)로 추출
print(df)

#토큰화: 한 단어씩 전부 쪼갬 | 사전 구축: 중복 제외, 고유한 단어들로 번호를 매겨 사전을 구축 | 수치화: 각 문장 별로 고유한 단어들이 존재하는 지를 확인할 수 있음(결과)
#기계 시점 출력 결과) 0인 첫번째 문장을 보면 and는 1 but은 0 <- 수치화를 나타냄 
#h_text안에 기재된 모든 단어들을 나열하여 각 문장 별로 기재된 단어들을 카운트