#모델이 얼마나 잘 맞췄는지(정확도)를 계산
import numpy as np
from sklearn.linear_model import LogisticRegression
#sklearn 안에 metrics(평가 지표)도구를 꺼내서 accuracy_score(정확성)을 나타내는 도구를 사용
from sklearn.metrics import accuracy_score

#환자 데이터들
X=np.array([
    [36.5],
    [36.9],
    [37.2],
    [37.6],
    [38.0],
    [36.7],
    [37.8]
])
#0=음성,1=양성
y = np.array([0,0,0,1,1,0,1])
#회귀(학습할 의사)기반 틀 만들기
model=LogisticRegression()
model.fit(X,y) #학습

print('모델 학습 완료')

user_temp=float(input('사용자의 체온을 입력:'))

user_data=np.array([[user_temp]])
pred_temp=model.predict(user_data)[0]#학습된 회귀(의사)기반 틀이 첫번째(사용자) 양성,음성 예측
proba_temp=model.predict_proba(user_data)[0][1]#사용자가 양성일 예측에 대한 가능성

print(f'발열(양성) 확룰:{proba_temp*100:.1f}')
#졍확성 나타내서 비교하기 
#y는 환자들의 결과 그리고 회귀(의사) 기반 틀이 학습한 걸 토대로 결과에 대한 정확성을 비교
pred_temp_model=model.predict(X)#학습
accuracy_model=accuracy_score(y,pred_temp_model)
print(f'정확도: {accuracy_model*100:.1f}%')