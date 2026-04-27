#본 EMR데이터는 XX대학교  극히 일부분 '틀'(내용X)을 가지고 진행했음을 알립니다
# 절대 실제 환자의 기록을 토대로 학습을 하지 않았으며 이를 가지고 유포하거나 타인에게 공유할 시, 국내 의료법에 위반된다는 걸 인지하고 있습니다.
#%%
import pandas as pd
import os
import pyreadstat 
import numpy as np
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
csv_edit=csv_load[['Gender','Age','WC','SBP','DBP','BMI','GLU','HbA']]

#성별에 따른 조건 부여(WC는 남,여 기준이 다름)
condition = [
    #Male
    (csv_edit['Gender']==1)&(csv_edit['WC']>=90.0)& #남자는 90
    (csv_edit['HbA']>=6.5)&(csv_edit['GLU']>=126.0)& # 당뇨 확진 수준
    (csv_edit['BMI']>=30.0)& #WC와 함께 복부비만
    (csv_edit['SBP']>=140.0)&(csv_edit['DBP']>=90.0),
    #Female
    (csv_edit['Gender']==0)&(csv_edit['WC']>=85.0)& #여자는 85
    (csv_edit['HbA']>=6.5)&(csv_edit['GLU']>=126.0)& # 당뇨 확진 수준
    (csv_edit['BMI']>=30.0)& #WC와 함께 복부비만
    (csv_edit['SBP']>=140.0)&(csv_edit['DBP']>=90.0)
]
csv_edit['positive_result']=np.select(condition,[1,1],default=0) #list를 가져와 조건을 전부 만족할 시 둘 다 1부여,그게 아니면 기본값인 0부여
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
# %%
