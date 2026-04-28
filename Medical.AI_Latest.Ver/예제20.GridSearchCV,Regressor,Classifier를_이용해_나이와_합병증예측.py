#본 EMR데이터는 XX대학교  극히 일부분 '틀'(내용X)을 가지고 진행했음을 알립니다
# 절대 실제 환자의 기록을 토대로 학습을 하지 않았으며 이를 가지고 유포하거나 타인에게 공유할 시, 국내 의료법에 위반된다는 걸 인지하고 있습니다.
#%%
import pandas as pd
import os
import pyreadstat 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
#Classifier: 양성,음성을 분류. Regressor:나이 예측
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,accuracy_score

#sav->csv (metadata거르기)
sav_file = 'training.sav'
csv_file = 'csv/[csv]Predict_info.csv' #training->[csv]Predict_info.csv 

try:
    if not os.path.exists(csv_file):
        sav_load,_=pyreadstat.read_sav(sav_file)
        print('Converting sav->csv..')
        sav_load.to_csv(csv_file,index=False,encoding='utf-8-sig')
        print('Converted!')
except Exception as e:
    print(f'Oops! {e}')

csv_load=pd.read_csv(csv_file)

#사용자가 모은 정보만 학습하기 위해 따로 변수 만들기(drop하기엔 카테고리가 너무 많음)
csv_edit=csv_load[['Gender','Age','WC','SBP','DBP','BMI','GLU','HbA']].copy()
#.copy():원본 데이터를 건드는 것에 예민한 파이썬이 경고를 낼 수 있기에, .copy()라는 복사본 인증 마크를 달아주면 된다

#성별에 따른 조건 부여(WC는 남,여 기준이 다름)
condition = [
    #Male
    (csv_edit['Gender']==1)&(csv_edit['WC']>=90.0), #남자는 90
    #Female
    (csv_edit['Gender']==2)&(csv_edit['WC']>=85.0) #여자는 85
]
wc_condition=np.select(condition,[1,1],default=0) #list를 가져와 조건을 전부 만족할 시 둘 다 1부여,그게 아니면 기본값인 0부여
risk_condition = (
    (csv_edit['HbA']>=6.5).astype(int)+ # astype으로 정수로 바꿔준 후 조건 부여
    (csv_edit['GLU']>=126.0).astype(int)+ # 당뇨 확진 수준
    (csv_edit['BMI']>=30.0).astype(int)+ #WC와 함께 복부비만
    (csv_edit['SBP']>=140.0).astype(int)+
    (csv_edit['DBP']>=90.0).astype(int)
)
total_risk = wc_condition+risk_condition
csv_edit['positive_result']=(total_risk >= 3).astype(int) #양성인 값 정수로 변환
# #사용자가 모은 정보만 학습하기 위해 따로 변수 만들기(drop하기엔 카테고리가 너무 많음)
# csv_edit=csv_load[['Age','WC','SBP','DBP','BMI','GLU','HbA']]
#train_test
X_rf = csv_edit[['Age','WC','HbA','GLU','BMI','SBP','DBP']]
Y_rf = csv_edit['positive_result']
Y_reg = csv_edit['Age'] #정답지로 reg 학습시키기
X_train,X_test,Y_train,Y_test,Y_train_reg,Y_test_reg=train_test_split(X_rf,Y_rf,Y_reg,test_size=0.2,random_state=42)
#나이예측(Reg)
X_train_reg = X_train.drop(columns=['Age'])
X_test_reg = X_test.drop(columns=['Age']) 

#doctor!
doctor_rf=Pipeline([
        ('simp',SimpleImputer(strategy='median')),
        ('stad',StandardScaler()),
        ('rf',RandomForestClassifier(random_state=42))
    ])
doctor_reg=Pipeline([
    ('simp',SimpleImputer(strategy='median')),
    ('stad',StandardScaler()),
    ('reg',RandomForestRegressor(random_state=42))
])
re_patient=csv_edit['positive_result'].value_counts()
for pat_name,patients in re_patient.items():
    if pat_name == 0:
        print(f'Negative patients: {patients}')
    elif pat_name == 1:
        print(f'Postive patients: {patients}')
# print(f'CSV patients result: {csv_edit['positive_result'].value_counts().to_string()}')
# value_counts(): 각 카테고리 별 몇 개인 지 확인할 수 있음
# to_string(): 판다스의 복잡한 기능을 아예 덜어주고 내용을 문자형으로 변환해 결과만 내주는 내장어
print('Auto factory started...')
#GridSearchCV, dictionary형태 유지
#RandomForestClassifier
rf_grid = {
    'rf__n_estimators': [100,150,200], #가장 최고의 n_estimators개수를 선택
    #결과 도출법: 설정한 추정자(나무)들이 100그루 중 70그루가 1이라고 하면 다수결 원칙에 의해 1로 내놓는다
    'rf__max_depth': [10,20,30] 
}#RandomForestRegressor
reg_grid = {
    'reg__n_estimators':[50,100,200], #가장 최고의 n_estimators개수를 선택
    #결과 도출법: MSE(Mean Squared Error)평군 제곱 오차를 써서 각 추정자들(나무)이 부른 숫자의 평균을 내는 방식이다
    #여기서 각 추정자가 부른 값이 너무 크면 제곱을 하여 오차가 엄청 커지게 된다(기하급수적으로 뛰는 평균 오차 방지)
    'reg__max_depth':[10,20,30]
}
main_rfgrid=GridSearchCV(
    estimator=doctor_rf, #추정할 모델 (모델 객체 지정)
    param_grid=rf_grid, # 딕셔너리로 주어진 값으로 테스트 (최고의 경우를 찾기)
    scoring = 'accuracy', #시험 체점 방식
    cv=5, #재차 확인
    n_jobs=-1 #코어 전부 총동원
)
main_reggrid=GridSearchCV(
    estimator=doctor_reg,
    param_grid=reg_grid,
    scoring='neg_mean_squared_error', #neg(ative)음수 평균 제곱 오차는 에러의 확률이 작아야 좋다
    cv=5,
    n_jobs=-1 
)
#학습
main_rfgrid.fit(X_train,Y_train)
main_reggrid.fit(X_train_reg,Y_train_reg)
#ML 마무리
try:
    for rfname,rfcontent in main_rfgrid.best_params_.items():
        print(f'RF_best_param) {rfname} : {rfcontent}')
    print('RF: Complete load')
except Exception as e:
    print(f'Oops! {e}')
try:
    for regname,regcontent in main_reggrid.best_params_.items():
        print(f'REG_best_param) {regname} : {regcontent}')
    print('REG: Complete load')
except Exception as e:
    print(f'Oops! {e}')
#RF
best_rfmodel = main_rfgrid.best_estimator_
best_rfscoring = main_rfgrid.best_score_
#REG
best_regmodel = main_reggrid.best_estimator_
best_regscoring = main_reggrid.best_score_
#MSE는 어떻게든 값을 작게 나타내려 해서 음수로 출력할 것이다
real_mse = best_regscoring * -1 #음수 출력 방지를 위한 대처1
rmse = np.sqrt(real_mse) #MSE는 오차를 제곱하는 식이 포함되어 있기에 sqrt(Square Root)로 제곱을 풀어준다->제곱이 안된 진짜 값
print('Total Model Performance')
print('-'*50)
print(f'Best RF Model: {best_rfmodel}')
print(f'RF Accuracy: {best_rfscoring*100:.1f}%')
print('-'*50)
print(f'Best REG Model: {best_regmodel}')
print(f"Regressor Model's 'Age' Mean Squared Error: {rmse:.2f} (If Value is low, This model is above average)")
print('-'*50)
print('All model loaded!')
print('-'*50)
#User
while True:
    print('Test User')
    def UserInfo(prompt,minval,maxval,is_float=True):
        while True:
            try:
                uvalue = float(input(prompt)) if is_float else int(input(prompt))
                if minval <= uvalue <= maxval:
                    print('entered')
                    return uvalue
                print('Over Range!')
            except ValueError:
                print('Oops! That type is wrong, try again')
    UserInfoGet = {
        'Gender':[UserInfo('What is your gender? (1:Male,2:Female)',1,2,is_float=False)],
        'Age':[UserInfo('How old are you? (Range:0~150)',0,150,is_float=True)],
        'WC':[UserInfo('Enter your Waist Circumference',40,150)],
        'SBP':[UserInfo('Enter your Systolic Blood Pressure',40,250)],
        'DBP':[UserInfo('Enter your Diastolic Blood Pressure',40,180)],
        'BMI':[UserInfo('Enter your Body Mass Index',10,70)],
        'GLU':[UserInfo('Enter your Glucose',50,170)],
        'HbA':[UserInfo('Enter your HbA(Hemoglobin A)',2,15)]
    }
    #Classifier
    user_dt=pd.DataFrame(UserInfoGet)
    rf_columns = ['Age', 'WC', 'HbA', 'GLU', 'BMI', 'SBP', 'DBP'] #강제로 학습 순서와 맞추기
    # del_gen=user_dt.drop(columns=['Gender']) #학습 카테고리에서 성별은 제외함
    del_gen=user_dt[rf_columns] #DF가 리스트 형태에서 똑같은 이름을 보고 매치
    pred_user=main_rfgrid.predict(del_gen)[0]
    proba_user=main_rfgrid.predict_proba(del_gen)[0][1]
    #Regressor
    reg_columns = ['WC', 'HbA', 'GLU', 'BMI', 'SBP', 'DBP']
    # del_age=user_dt.drop(columns=['Gender','Age']) #마찬가지로 성별 제외하고 나이도 같이 제외
    del_age=user_dt[reg_columns]
    pred_age=main_reggrid.predict(del_age)[0]
    #result
    print('-'*50)
    print('Successfully Diagnosis!')
    print(f"AI predicted User's age: {pred_age:.0f} old")
    if pred_user == 1:
        print(f'Result: Positive Probability{proba_user*100:.1f}')
        print('Care your Body and GO to hospital, AI recommend consulting with doctor')
    elif pred_user == 0:
        print(f'Result: Negative Probability{(1-proba_user)*100:.1f}')
        print('Good Condition! Keep your healthy body!')
    #subplot:(큰 도화지를 반으로 나눠서<-사용자가 별도로 설정) 한 번에 띄우는 기능
    fig, axes = plt.subplots(1,2,figsize=(14,6)) #fig: 전체 액자 프레임,figsize:액자크기(가로 세로),(1줄 2칸)
    #axes:그림을 그리는 공간(axes[0],axes[1]로 나눠서 counplot,scatterplot으로 나눠야 함)
    #전체 환자 양성,음성 표
    sns.countplot(x='positive_result',data=csv_edit,ax=axes[0]) #axes[0]:왼쪽에다가 그리기
    axes[0].set_title('Postive|Negative Patients Result') #set을 써서 어떤 방(axes[0],[1])에 들어갈 지를 지정해주는 코드
    axes[0].set_xticks([0,1],['Negative(0)','Positive(1)']) #0과 1에 이름 붙여주기
    #전체 환자 중 사용자를 띄운 표
    sns.scatterplot(x='Age',y='GLU',marker='o',color='gray',data=csv_edit,label='Other Patients',alpha=0.5,ax=axes[1]) #다른 환자들
    #alpha:점의 투명도를 0(완전 투명)에서 1(불투명)사이로 조절->겹쳐있는 점을 세세하게 보기
    u_age=user_dt['Age'][0] #사용자 나이 가져오기
    u_glu=user_dt['GLU'][0] #사용자 글루코스 가져오기
    #모든 환자의 데이터를 불러올 것은 아니기에 plt.scatter로 사용자의 정보 하나만 가져오기
    axes[1].scatter(x=u_age,y=u_glu,marker='*',color='red',label='User',zorder=5)#행동 명령일 때는 set_을 쓰지 않는다(legned,show..)
    #zorder: 특정 정보를 맨 앞으로 가져오거나 맨 뒤로 보내는 기능
    axes[1].set_title('User Status')
    axes[1].legend() #안내표
    plt.tight_layout() #그림이 겹치지 않게 간격을 조절해주는 도구
    plt.show()
    while True:
        YesNo=input('Can you restart this test? (press y|n)')
        if YesNo == 'y':
            break
        elif YesNo == 'n':
            break
        else:
            print('You can only enter two spell(y|n)')
    if YesNo == 'y':
        print('Countinue test')
    elif YesNo == 'n':
        print('End test')
        break
# %%
