import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm #탐지견 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#글짜
mpl.rcParams['font.family'] = 'AppleGothic'
mpl.rcParams['axes.unicode_minus'] = False 

np.random.seed(42) #이따 확인할 것 
r_samples=300

#정보 
r_age=np.random.randint(0,100,r_samples)
r_bmi=np.random.uniform(14.5,50.5,r_samples)
r_sysbp=np.random.randint(80,220,r_samples)

r_risk=[]
for r in range(r_samples):
    if r_bmi[r] >= 30 and r_sysbp[r] >= 150:
        r_risk.append(1)
    else:
        r_risk.append(0)

p_df = pd.DataFrame({
    "age":r_age,
    "bmi":r_bmi,
    "sysbp":r_sysbp,
    "risk":r_risk
}) #행렬의 기억을 더듬어봐라.. p

X=p_df[["age","bmi","sysbp"]] #p_df 써야지
Y=p_df["risk"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=42) #이따 확인할 것

doctor=make_pipeline(
    StandardScaler(),
    SVC(probability=True,
    kernel='rbf'))
#kernel:'커널 트릭' 2d에서 3d로 붕 띄워 올린 다음 기괴한 형태의 경계선도 거침없이 그려냄
#->그러나 너무 무거운 기술이기에 엉뚱한 오진을 남길 수 있으니 잘 써야 함

doctor.fit(X_train,Y_train)

pred_data=doctor.predict(X_test)
acc_data=accuracy_score(pred_data,Y_test)

print(f'Model Accuracy: {acc_data*100:.2f}%')

while True: 
    def get_value(prompt,min_val,max_val,is_float=False):
        while True:
            try:
                u_value=float(input(prompt)) if is_float else int(input(prompt))
                if min_val <= u_value <= max_val:
                    return (u_value) 
                print('Over Range, Again enter value')
            except ValueError:
                print('Type is Error, Edit Type!')
    user_info={
    "age":[get_value('Enter Age:',0,100)],
    "bmi":[get_value('Enter BMI value:',14.5,50.5,is_float=True)],
    "sysbp":[get_value('Enter Systolic Blood Pressure value:',80,220)] #Dictionary 사용/DataFrame 묶기 #열에 사용자의 값을 추가하기 위해 지정.
    }
    new_data=pd.DataFrame(user_info)
    new_pred=doctor.predict(new_data)[0]
    new_proba=doctor.predict_proba(new_data)[0][1]
    print(f'User Information| Age:{user_info["age"]}, BMI:{user_info["bmi"]}, Systolic Blood Pressure:{user_info["sysbp"]}') #사전에서 사용자 값을 빼고 싶으면 사전명을 가져와 빼고 싶은 값을 빼면 됨
    if new_pred == 0:
        print('User status is Negative(0). That mean Your healthy is above average ')
        print(f'Negative Predict Probability: {(1-new_proba)*100:.2f}%')
    elif new_pred == 1:
        print('User status is Positive(1). That mean Your healthy is not good ')
        print(f'Positive Predict Probability: {new_proba*100:.2f}%')
    else:
        print('Model can not measure this situation')
        print(f'Predict Probability: {new_proba*100:.2f}%')
    plt.scatter(p_df["age"],p_df["bmi"],color='blue',marker='o',label='Other Users')
    plt.scatter(user_info["age"], user_info["bmi"],color='red',marker='*',label='User') #우선 혈압을 빼고 나이와 BMI만으로 (X,Y)
    plt.xlabel('AGE')
    plt.ylabel('BMI')
    plt.legend() # 라벨 표시
    plt.show()
    while True:
        user=input('Can you restart this test? (Press y|n)')
        if user == 'y':
            break
        elif user == 'n':
            break
        else:
            print('You can only enter two spell (y|n)')
    if user == 'y':
        print('Countinue Test')
    elif user == 'n':
        print('End Test')
        break


