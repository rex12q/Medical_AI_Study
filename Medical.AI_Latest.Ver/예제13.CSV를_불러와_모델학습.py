#본 예제는 예제_8를 기반으로 코드가 구성되고 돌아간다.
import pandas as pd
#ensemble: 조화 | RandomTreeClassifier: 정확도 성능(black-box)에 초점을 둔 모델 그러나 설명 부분에서 효율이 떨어짐
#정확도 성능에 올인을 하다 보니 결과 예측 모델에서 정말 많이 쓰이지만 그에 따른 단점은 항상 뒤 따른다
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#csv 파일 불러오기 
df = pd.read_csv('My first patients.csv')
#csv를 불러올 때는 pd 내장어인 read_csv를 이용해서 불러올 수 있다.

#csv파일에는 diabetes_target(당뇨 판정[0|1])이 없기에 for문으로 누적 시스템을 셋팅
risk_data=[]
for t in range(len(df)): #df의 형식은 거대한 pd표이다 그러므로 len()으로 나타내서 쓰자
    if 70 <= df["glucose"][t] <= 200 and df["bmi"][t] >= 25.5: #예제8을 보면 "glucose"라 표시가 되어있기에 df[]그대로 가져옴.
        #Glucose: 70~200이상
        #배열 개념을 생각해보자.. []<-이걸로 받기
        risk_data.append(1)
    else:
        risk_data.append(0)

#csv에는 당뇨 판정이 없으니 같이 합치기
df['diabetes_target']=risk_data
#이 부분이 상당히 헷갈림| 같은 변수명을 써서 진행을 하면 df가 합쳐진 상태에서가 아닌 자체적으로 risk_data만 drop을 하는 것인데 이는 컴퓨터도, 사람도 못 알아먹기에 
#이렇게 쓰는 것을 추천

#X와 Y를 분리해서 train과 test가 학습할 수 있도록 도와주자
X=df.drop(columns=["patient",'diabetes_target']) #for문으로 만든 당뇨판정 열을 'diabetes_target'으로 설정 
#주의할 점은 "patient"는 따로 논리회귀 공식에 계산이 안된다는 것이다. 그래서 drop을 하고 가야 함
#"patient"는 X에 추가하자

y=df['diabetes_target'] #Y는 정답지로 두자

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42) #문제,실제 시험지, 정답지 준비
#항상 같은 방법으로 섞도록 random_state=42로 고정

#Studying
mini_doctor=make_pipeline(
    StandardScaler(),
    RandomForestClassifier()
    )
#glucose의 체급을 평등하게 맞춰주기 위해 StandardScaler를 사용/원활한 모델 사용을 위해 make_pipeline을 사용
mini_doctor.fit(X_train,y_train)

#result
print('학습이 성공적으로 마무리 됐습니다.')
print(f'미니 의사의 점수(?): {mini_doctor.score(X_test,y_test):.2f}') #점수를 나타내고 싶으면 내장어인 score를 써서 채점해보자!

#glucose: 혈당(피 속에 녹아있는 포도당)
#나이랑 bmi가 높을 수록 혈당이 몸에 미치는 악영향은 배가 된다는 사실
#당뇨병: 무언가를 섭취하면 포도당으로 에너지가 남게되는데, 인슐린 문제로 조절이 안되, 핏 속에 너무 달달하게(많이) 남아있는 병

