#의료AI의 시작은 대부분 데이터 정리(pandas)임
#load CSV (액셀 스프레드시트 파일을 불러오는 연습)
#pandas: 의료검진 표(액셀 같은 표 데이터)를 다루는 도구 
import pandas as pd

#가짜용 데이터
fake_data={
    "patient": ["A","B","C","D","E"], #환자 
    "glucose": [85, 95, 110, 140, 100],   # 혈당
    "bmi":     [20.1, 22.0, 25.5, 30.2, 24.0] #수치
}

fake_data_frame = pd.DataFrame(fake_data) #판다가 쓰는 데이터 프레임, 틀 안에 가짜용 데이터를 통해 행렬 표를 만듦
print(fake_data_frame) #행렬표 

print("혈당 평균:", fake_data_frame["glucose"].mean())
print("혈당 분산:", fake_data_frame["glucose"].var())
print("혈당 최대:", fake_data_frame["glucose"].max())
print("혈당 최소:", fake_data_frame["glucose"].min())

high=fake_data_frame[fake_data_frame["glucose"] >= 126]
print('\n공복혈당 126 이상 (예시 기준) 환자')
print(high)

#이제 CSV로 추출을 해보자
fake_data_frame.to_csv('My first patients.csv', index=False) 
#.to_csv는 사용자가 원하는 프레임을 가지고 csv를 출력하는 내장어이다.
in#dex는 인덱스 번호할 때 그 인덱스가 맞다 보통 판다스가 표를 만들 때, 왼쪽에 0,1,2,3..하고 자동으로 줄 번호(인덱스)를 매겨준다.
#만약 index=Fasle를 안 쓰면 나중에 파일을 불러올 때 '쓸데없는 줄 번호'가 새로운 데이터 열'인 것 처럼 
#액셀에 같이 저장되어 버린다(patient,glucose,bmi도 한 행에 포함이 된다 그러면 컴퓨터가 '얘도 데이터에 포함되나?'싶어서 0을 부여한다 이 때 부터 데이터는 꼬인다)
#그래서 사용자가 남긴 데이터 열만 남겨두고 컴퓨터가 자동으로 생성한 번호들을 제외하라는 명령어가 index=False
print('성공적으로 파일 생성이 됐습니다.')
