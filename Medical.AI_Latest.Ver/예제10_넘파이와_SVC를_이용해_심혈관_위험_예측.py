print('성인 기준: 나이와 bmi, 혈압으로 심혈관 위험을 예측해보자')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split #시험지, 정답지
from sklearn.svm import SVC  #support vector machine
from sklearn.metrics import accuracy_score #정확성 

#무작위 100명 만들기
np.random.seed(42) # 언제 돌려도 일관된 순서로
p_samples=100 # 사람 100 설정
p_patient=1
n_patient=0


d_age=np.random.randint(20,80,100) #정수
d_bmi=np.random.uniform(18,35,100) #실수
d_sysbp=np.random.randint(110,180,100) 
#누적 시스템
all_risk=[]
for i in range(p_samples):
    if d_bmi[i] >= 25 and d_sysbp[i] >= 140:
        all_risk.append(1)
    else:
        all_risk.append(0)

#DF
all_df=pd.DataFrame({
    "a_age":d_age,
    "a_bmi":d_bmi,
    "a_sysbp":d_sysbp,
    "a_risk":all_risk,
    })
#시험지와 정답지 (train_test_split)
X=all_df[["a_age", "a_bmi", "a_sysbp"]] #2차원배열
Y=all_df["a_risk"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=42)

#학습
p_model=SVC(probability=True)#확률
p_model.fit(X_train,Y_train)

#정확성
pred_test=p_model.predict(X_test) #실제 문제 풀기 
acc_score=accuracy_score(Y_test,pred_test)
print(f'모델 정획성: {acc_score*100:.2f}%')
print('해당 모델은 성인을 기준으로 학습을 했습니다.')

#사용자 정보
while True:
    while True:
        age=int(input('나이를 입력해주세요'))
        if 20<= age <= 140: 
            print('올바른 나잇값이 기입됐습니다.')
            break
        else: 
            print('20살부터 140살까지 기입이 가능합니다.')

    while True:
        weight=float(input('몸무게를 입력해주세요'))
        height=float(input('키를 입력해주세요'))
        if 0<=weight<=200 and 0<=height<=300:
            print('올바른 키와 몸무게가 기입됐습니다.')
            break
        else:
            print('몸무게는 200kg까지, 키는 300cm까지 기입이 가능합니다.')

    while True:
        sys_bp=int(input('수축기 혈압을 입력해주세요'))
        if 40<=sys_bp<=180:
            print('올바른 값이 기입됐습니다.') 
            break
        else:
            print('40부터 180까지 기입이 가능합니다.')

    #bmi
    height_m=height/100
    bmi_value=weight/(height_m**2)
    user_bmi=round(bmi_value, 2)

    #예측 가능성
    new_data=pd.DataFrame([[age,user_bmi,sys_bp]], columns=["a_age", "a_bmi", "a_sysbp"]) #DF 셋팅 후(row), 열 이름 붙히기(column,동시에 묶기)
    pred_data=p_model.predict(new_data)[0] #사용자 인덱스
    proba_data=p_model.predict_proba(new_data)[0][1] #양성 인덱스 

    #결과
    if p_patient > n_patient:
        print('모델 학습 결과: 값은 양성 환자가 더 많습니다.')
    else:
        print('모델 학습 결과: 값은 음성 환자가 더 많습니다.')

    if pred_data == 0 and bmi_value <= 25 and sys_bp <=140:
        print('사용자의 결과는 음성입니다')
    else:
        print('사용자의 결과는 양성입니다')
        print(f'사용자가 양성일 가능성 {proba_data*100:.2f}%')
    
    #재실행 여부
    again_answer=input('검사를 다시 진행하겠습니까?(y/n)')
    if again_answer == 'n':
        break
