print('나이와 bmi, 혈압으로 심혈관 위험을 예측해보자')
import numpy as np #판다 DF가 있기에 안 쓰지만, 기준을 정해야기에 설정.
from sklearn.model_selection import train_test_split #학습용과 정답지
from sklearn.svm import SVC # support vector machine
import pandas as pd
from sklearn.metrics import accuracy_score #정확성

#랜덤 숫자를 만들 때 기준 고정, 언제 실행해도 같은 데이터가 나오게 하기 위해서임
np.random.seed(42)
#샘플 100명 (설정)
n_samples = 100

#랜덤화 과정 
age=np.random.randint(20,81,100) #20살부터 80살까지 무작위의 100명 뽑기 
bmi=np.random.uniform(18,35,n_samples) #수치, 여기서 uniform은 float을 무수히 뽑겠다는 의미.
sysbp=np.random.randint(110,180,n_samples) #수축기 혈압
#(파이썬 기본)random.randint는 정수 1개를 무작위로 선택
#(파이썬 넘피)오른쪽(끝)값은 범위에 포함하지 않고, 넘피는 ML에 최적화 된 도구이기에 한꺼번에 무작위의 숫자를 뽑아낸다

#위험 요소 설정(누적 시스템)
risk=[] #이 칸에 0또는 1을 계속 넣을(누적 시스템) 예정
for i in range(n_samples): #i를 0부터 99까지 바꾸면서 반복
    if sysbp[i] >= 140 or bmi[i] >= 28:
        risk.append(1)#1나타내기
    else:
        risk.append(0)

#pd.Df형식 "설정할 이름":변수명/같은 데이터셋에 있기에 ','필수
all_df=pd.DataFrame({
    "age":age,
    "bmi":bmi,
    "sysbp":sysbp,
    "risk":risk
})

#여러 개의 열을 뽑아서 DF형태 유지
X=all_df[['age','bmi','sysbp']]#X:특징
Y=all_df['risk']#하나의 리스트만 있기에 1차원 배열, Y:정답

#변수명 설정 후 테스트 사이즈
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=42)
#X_train,Y_train: X 연습'학습지' 75%, Y '연습'정답지' 
#X_test,Y_test: X 실제 시험 '문제지' 25%, Y 실제 시험 '정답지'

db_model=SVC(probability=True)
db_model.fit(X_train,Y_train)

#정확성
pred_model=db_model.predict(X_test) #진짜 문제 풀게 하기(예측) 
acc_data=accuracy_score(Y_test,pred_model) #푼 문제와 정답 비교 
print(f'모델의 정확성{acc_data*100:.2f}%')

#사용자 데이터
print("\n--- 사용자 데이터 입력 ---")
u_age=int(input('나이 입력'))
u_bmi=float(input('bmi 입력'))
u_sysbp=int(input('수축기 혈압 입력'))

#예측 가능성
new_u_data=pd.DataFrame([[u_age,u_bmi,u_sysbp]], columns=['age', 'bmi', 'sysbp'])
new_pred=db_model.predict(new_u_data)[0]#사용자 인덱스
new_proba=db_model.predict_proba(new_u_data)[0][1]#사용자, 양성 인덱스

#결과
print(f'\n--- 예측 결과 ---')
if new_pred == 1 and new_proba*100 >= 50:
    print('양성')
    print(f'양성 가능성{new_proba*100:.2f}')
else:
    print('음성')
    print(f'양성 가능성{new_proba*100:.2f}')


#추가로 썼던 코드들
#음성 데이터 50명 데이터 셋
#neag_df=pd.DataFrame({
#    "age": np.random.randint(20,52,50), #나이 
#    "bmi": np.random.uniform(18,25,50), #수치 
#    "sysbp": np.random.randint(110,140,50), #수축기 혈압
#    "risk": 0   # 0=안전
#})

#양성+음성 데이터 셋
#concat:데이터 합치기 (sql)
#all_df=pd.concat([pos_df,neag_df]).reset_index(drop=True)
#.reset_index(drop=True) 인덱스(번호) 다시 0~99로 정리
#합치면 인덱스가 꼬일 수 있음(AI 학습은 깔끔한 인덱스가 좋음)