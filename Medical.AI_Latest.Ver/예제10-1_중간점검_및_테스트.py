import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd 
from sklearn.metrics import accuracy_score 

np.random.seed(42)
p_samples=100

db_age=np.random.randint(0,80,p_samples)
db_bmi=np.random.uniform(17.5,40,p_samples)

all_risk=[]
for i in range(p_samples):
    if db_bmi[i] >= 25:
            all_risk.append(1)
    else:
            all_risk.append(0)

db_data=pd.DataFrame({
'age':db_age,
'bmi':db_bmi,
    'risk':all_risk
})
X=db_data[['age', 'bmi']]
y=db_data['risk']

doctor=LogisticRegression()
doctor.fit(X,y)

acc_pred=doctor.predict(X)
acc_data=accuracy_score(y,acc_pred)

print(f'{acc_data*100:.2f}%의 정확성을 가짐')
print('모델 학습 완료')
while True:
    u_age=int(input('나이 입력'))
    u_bmi=float(input('bmi 입력'))

    new_data=pd.DataFrame([[u_age,u_bmi]], columns=['age', 'bmi'])
    new_pred=doctor.predict(new_data)[0]
    new_proba=doctor.predict_proba(new_data)[0][1]

    print(f'사용자 정보/ 나이:{u_age}, bmi:{u_bmi}')
    if u_bmi >= 25:
        print('사용자는 진료가 필요합니다.')
    else:
        print('사용자는 이상이 없습니다.') 
    
    again_answer=input('테스트를 다시 진행하겠습니까? (y/n)')
    if again_answer != 'y':
        break
