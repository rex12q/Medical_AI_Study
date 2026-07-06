#본 EMR데이터는 XX대학교  극히 일부분 '틀'(내용X)을 가지고 진행했음을 알립니다
# 절대 실제 환자의 기록을 토대로 학습을 하지 않았으며 이를 가지고 유포하거나 타인에게 공유할 시, 국내 의료법에 위반된다는 걸 인지하고 있습니다.
#%%
import numpy as np
import pandas as pd
import pyreadstat as pt
import shap
import os
from sklearn.model_selection import train_test_split,GridSearchCV
from xgboost import XGBClassifier #y피처를 분류형으로
from sklearn.preprocessing  import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
#C_R: 정확도,정밀도,f1,재현율 전부 출력하여 한 눈에 볼 수 있음
from sklearn.metrics import classification_report 
#path
# sav_path='Data.sav'
csv_path='[csv]XGBoost_test.csv'

#remind part
#convert and load 
# try:
#     if not os.path.exists('csv_path'):
#         sav_load,_=pt.read_sav(sav_path)
#         print('sav->csv...')
#         sav_load.to_csv(csv_path,index=True,encoding='')
# except Exception as e:
#     print(f'{e}')

#metadataonly
# try:
#     _,meta=pt.read_sav(sav_path)
#     for c_name,c_label in meta.column_names_to_labels:
#         label=c_label if c_label else ''
#         print(f'{label}')
# except Exception as e:
#     print(f'{e}')

#load csv
csv_df=pd.read_csv('[csv]XGBoost_test.csv')

#기준 맞추기 (다중 조건 = select)
if 'WC' in csv_df.columns and 'Gender' in csv_df.columns:
    gender_wc= [
        (csv_df['Gender'] == 1) & (csv_df['WC'] >= 90),
        (csv_df['Gender'] == 2) & (csv_df['WC'] >= 85)
    ]
    info_wc=np.select(gender_wc,[1,1],default=0)
else:
    info_wc=0 #col이 없을 경우 방어코드
risk_col=(
    (csv_df['GLU'] >= 126.0).astype(int)+
    (csv_df['HbA']>=6.5).astype(int)+
    (csv_df['BMI']>=30.0).astype(int)+
    (csv_df['SBP']>=140.0).astype(int)+
    (csv_df['DBP']>=90.0).astype(int)
)
total_risk_col=info_wc+risk_col
csv_df['HIGH_RISK']=(total_risk_col >= 3).astype(int) #위험항목 3개일 경우 1
#merge data
num_col=['GLU','HbA','BMI','SBP','DBP','WC']
stad_pp=Pipeline([('stad',StandardScaler())]) #표쥰화 스케일링 진행
total_col=ColumnTransformer(
    transformers=[
        ('stad_pp',stad_pp,num_col) #모듈 이름,대상
    ]
)
#divide,load data
X_cla=csv_df[['Age','GLU','HbA','BMI','SBP','DBP','WC']]
y_cla=csv_df['HIGH_RISK']
X_train,X_valid,y_train,y_valid=train_test_split(X_cla,y_cla,test_size=0.2,random_state=42)

doctor=Pipeline([
    ('total',total_col),
    #multi:softprob| 결론만 도출하는 것이 아닌 각 클래스(0,1..)에 얼마나 해당되는 지를 알려주는 확률을 알려줌 (softmax 함수를 구해서 이용)
    ('xgb',XGBClassifier(random_state=42,objective='multi:softprob'))
])

#parameter tuning
para_tune={
    'xgb__n_estimators':[100,200,300,400,1000],
    'xgb__max_depth':[3,5,7,10,15],
    'xgb__learning_rate':[0.01,0.1],
    'xgb__subsample':[0.5,0.8,1.0] #데이터 전체를 외우는 꼼수 차단
}
model_tune=GridSearchCV(
    estimator=doctor,
    param_grid=para_tune,
    scoring='recall-macro', #False Negative (False Positive 약간의 오차가 있더라도 무조건 잘못된 FN은 바로 잡음)
    cv=6,
    n_jobs=-1 
)
#학습 및 예측, 결과
doctor.fit(X_train,y_train)
best_model=model_tune.best_estimator_
best_scoring=model_tune.best_score_

pred_col=best_model.predict(X_valid) #확률이 가장 높은 클래스 최종 출력
score_col=classification_report(
    y_valid,pred_col,
    target_names=['False(0)','Positive(1)'], #타겟 이름 설정(0,1,2)
    digits=3, #소수점 자리
    output_dict=True #Dict형태로 출력
        ) 
#SHAP(XAI)