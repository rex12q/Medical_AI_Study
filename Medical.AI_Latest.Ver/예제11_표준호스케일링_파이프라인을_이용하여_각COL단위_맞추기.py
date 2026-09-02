import numpy as np
import pandas as pd
#pipeline 이라는 상자에서 꺼낸 make_pipeline은 매번 과정을 따로 코드 짜기 귀찮아서 자동 공정(파이프라인)으로 묶어줌
from sklearn.pipeline import make_pipeline
#preprocessing 이라는 상자에서 꺼낸 StandardScaler는 데이터 체급을 맞추는 도구다. 모든 수치를 평균0, 표준편차1로 강제로 설정 ('같은 체급' 설정)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

np.random.seed(42)
p_samples=100

db_age=np.random.randint(0,100,p_samples)
db_bmi=np.random.uniform(17.5,35.5,p_samples)

risk=[]
for i in range(p_samples):
    if db_bmi[i] >= 25.6:
        risk.append(1)
    else:
        risk.append(0)

db_df=pd.DataFrame({
    "age":db_age,
    "bmi":db_bmi,
    "risk":risk
})

#db_df꺼 가져오기
X=db_df[['age','bmi']]
y=db_df['risk']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

#스마트 모델 만들기
doctor=make_pipeline( #1.매끄럽게 파이프라인으로 연결하기 
    StandardScaler(), #2.모든 값에 같은 가중치를 둘 것 (체급 맞추기)
SVC(probability=True) #3.SVC모델에 넣고 학습시키기
)
doctor.fit(X_train,y_train)

acc_pred=doctor.predict(X_test)
acc_data=accuracy_score(y_test,acc_pred)

print(f'모델의 정확도: {acc_data*100:.2f}%')
print('학습 완료')

while True:
    while True:
        u_age=int(input('나이 입력'))
        if 0 <= u_age <= 100:
            break
        else:
            print('올바른 값을 입력하시오')
    while True:
        u_bmi=float(input('bmi 입력'))
        if 0 <= u_bmi <= 60:
            break
        else:
            print('올바른 값을 입력하시오')
    new_data=pd.DataFrame([[u_age,u_bmi]], columns=["age","bmi"])
    new_pred=doctor.predict(new_data)[0]
    new_proba=doctor.predict_proba(new_data)[0][1]
    
    print(f'사용자 정보| 나이: {u_age}, bmi: {u_bmi}')
    if new_pred == 0:
        print('음성')
    else:
        print('양성')
        print(f'양성 확률: {new_proba*100:.2f}')
    while True:
        again=input('테스트를 다시 하겠습니까? (y/n)')
        if again == 'y':
            break
        elif again == 'n':
            break
        else:
            print('y/n 둘 중 하나만 입력하시오')
    if again == 'y':
        print('테스트 재개') 
    elif again == 'n':
        print('종료')
        break
