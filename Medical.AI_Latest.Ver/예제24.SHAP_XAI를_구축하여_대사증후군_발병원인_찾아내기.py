#본 EMR데이터는 XX대학교  극히 일부분 '틀'(내용X)을 가지고 진행했음을 알립니다
# 절대 실제 환자의 기록을 토대로 학습을 하지 않았으며 이를 가지고 유포하거나 타인에게 공유할 시, 국내 의료법에 위반된다는 걸 인지하고 있습니다.
#%%
import numpy as np
import pandas as pd
import pyreadstat as pt
import shap
import os
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from xgboost import XGBClassifier #y피처를 분류형으로
from sklearn.preprocessing  import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
#C_R: 정확도,정밀도,f1,재현율 전부 출력하여 한 눈에 볼 수 있음
#C_M: 모델이 어디서 컨닝하고 틀렸는지 직접 확인할 수 있음(시각화 자료 heatmap과 함께 참고함)
from sklearn.metrics import classification_report, confusion_matrix
#path
# sav_path='Data.sav'
csv_path='csv/[csv]XGBoost_test.csv'

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
csv_df=pd.read_csv(csv_path)

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
num_col=['GLU','HbA','BMI','SBP','DBP','WC'] #Age 칼럼은 표준화 스케일링 X
stad_pp=Pipeline([('stad',StandardScaler())]) #표쥰화 스케일링 진행
total_col=ColumnTransformer(
    transformers=[
        ('stad_pp',stad_pp,num_col) #모듈 이름,대상
    ],
    remainder='passthrough' #선택을 못받은 나머지 컬럼은 원본 파일에서 drop을 하지 않고 그대로 passthrough를 진행하여 영향을 받지 않고 살림
)
#divide,load data
X_cla=csv_df[['Age','GLU','HbA','BMI','SBP','DBP','WC']]
y_cla=csv_df['HIGH_RISK']
X_train,X_valid,y_train,y_valid=train_test_split(X_cla,y_cla,test_size=0.2,random_state=42)

doctor=Pipeline([
    ('total',total_col),
    #multi:softprob| 결론만 도출하는 것이 아닌 각 클래스(0,1..)에 얼마나 해당되는 지를 알려주는 확률을 알려줌 (softmax 함수를 구해서 이용)
    #그러나 위 채점 방식은 다중 분류 결과만 취급하기에 본 모델에선 이진 분류 채점 방식인 'binary:logistic'을 써야 함
    ('xgb',XGBClassifier(random_state=42,objective='binary:logistic')) #softprob->binary:logistic
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
    scoring='recall_macro', #False Negative (False Positive 약간의 오차가 있더라도 무조건 잘못된 FN은 바로 잡음)
    cv=6,
    n_jobs=-1 
)
#학습 및 예측, 결과
model_tune.fit(X_train,y_train) #doctor가 아닌 파라미터 튜닝이 진행된 model_tune으로 학습
best_model=model_tune.best_estimator_ #그래야 최고 추정자 수를 나타낼 수 있음
best_scoring=model_tune.best_score_

pred_col=best_model.predict(X_valid) #확률이 가장 높은 클래스 최종 출력
score_col=classification_report(
    y_valid,pred_col,
    target_names=['False(0)','Positive(1)'], #타겟 이름 설정(0,1,2)
    digits=3, #소수점 자리
    output_dict=True #Dict형태로 출력
        )
#C_M(혼동 행렬)
c_m=confusion_matrix(y_valid,pred_col) #(검증, 예측 데이터)
#종합 성적표 출력
print(score_col)

#SHAP(XAI)
plt.rc('font',family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False #axes유니코드 누락 방지 (일반 하이픈 사용)

#SHAP - 전처리가 끝난 데이터, 알맹이 엔진(XGBoost) | ColumnTransformer 해체
pre_step=best_model.named_steps['total'] #Stad,num_col 분류 작업
xgb_engine=best_model.named_steps['xgb'] #xgb 엔진 분류 작업

#X_valid 데이터 전처리 통과 작업
X_valid_pre=pre_step.transform(X_valid)
#검증 데이터 전처리 진행 후 칼럼 항목 작성(passthrough를 썼기에 Age칼럼은 맨 뒤로)
final_features=['GLU','HbA','BMI','SBP','DBP','WC','Age']
X_valid_df=pd.DataFrame(X_valid_pre,columns=final_features)
explainer=shap.TreeExplainer(xgb_engine)#TreeExplainer: 앙상블 기법인 모듈은 전부 쓸 수 잇음 (초고속 설명기)
#앙상블,트리 기반 엔진은 내부가 복잡하기에 각 행(환자)이(가) 왜 그렇게 진단을 받았는 지를 알기 위해, 각 칼럼 항목에 점수를 부여해줌
#구조: 환자(행), 피처 칼럼 항목(열), 진단 결과(차원)
shap_values=explainer(X_valid_df) #DF로 이뤄진 검증 데이터를 위와 같은 칼럼 부여 점수와 함께 행렬로 나타낸다

#시각화 자료 결과
#전체 EMR, 대사증후군 주요 원인은?
plt.figure(figsize=(10,6)) #row:1 col:2 
shap.summary_plot(shap_values,X_valid_df,show=False) #1.shap_values 2.원본 DF 3.이 외(show=False는 plt으로 수정하고 싶을 때 일시정지를 위한 용도)
#[:,:,2] <-다중 분류일 경우 ':'전체,전체,'2'진단 결과 | 이진 분류일 경우
plt.title('대사증후군 발병원인 (XAI)',pad=15) #시각화 자료, 글이랑 간격 띄우기
plt.tight_layout() #설정된 규격에 맞게 출력이 되도록 설정

#C_M을 통해 히트맵 출력 (컨닝,오류 확인) 
plt.figure(figsize=(10,6))
sns.heatmap(c_m,annot=True,fmt='d',cmap='Blues',
            xticklabels=['예측: 정상','예측: 고위험'],
            yticklabels=['실제: 정상','실제: 고위험'])
plt.title('혼동 행렬 결과',pad=15)
#annotation:주석 | 각 칸에 주석을 강제 삽입 (True)
plt.tight_layout() #설정된 규격에 맞게 출력이 되도록 설정
plt.show()


# %%
