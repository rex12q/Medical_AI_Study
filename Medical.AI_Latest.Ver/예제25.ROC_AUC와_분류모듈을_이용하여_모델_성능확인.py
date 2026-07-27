#본 EMR데이터는 XX대학교  극히 일부분 '틀'(내용X)을 가지고 진행했음을 알립니다
# 절대 실제 환자의 기록을 토대로 학습을 하지 않았으며 이를 가지고 유포하거나 타인에게 공유할 시, 국내 의료법에 위반된다는 걸 인지하고 있습니다.
import pandas as pd
import numpy as np
import seaborn as sns
import pyreadstat as pt 
import os
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import f1_score #(재현율,정밀도)
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

sav_file='Data.sav'
csv_file='csv/sample2.csv' #변경됨

#글꼴
plt.rcParams['font.family']='AppleGothic'
plt.rcParams['axes.unicode_minus']=False

#sav->csv
try:
    if not os.path.exists(csv_file):
        sav_load,_=pt.read_sav(sav_file) #무거운 메타데이터 거름
        print('Converting')
        sav_load.to_csv(csv_file,index=False)
        print('Converted')
except Exception as e:
    print(f'{e}')

#csv_load
csv_load=pd.read_csv(csv_file)

#만약 메타데이터를 불러와야 한다면
#metadataonly
try:
    _,meta=pt.read_sav(sav_file,metadataonly=True)
    for name,col in meta.column_names_to_labels.items():
        labels=name if col else 'None'
        print(f'{labels}')
except Exception as e:
    print(f'{e}')

#조건1 
if 'WC' in csv_load.columns and 'Gender' in csv_load.columns:
    q_wc=[
        (csv_load['Gender'] == 1) & (csv_load['WC'] >= 90),
        (csv_load['WC'] == 2) & (csv_load['WC'] >= 85)
    ]
    #다중 조건
    info_gen=np.select(q_wc,[1,1],default=0)
else:
    #방어책
    info_gen=0
#조건 1-2. 상세 조건
q_detail=(
    (csv_load['GLU'] >= 126.0).astype(int)+
    (csv_load['HbA'] >= 6.5).astype(int)+
    (csv_load['BMI'] >= 30.0).astype(int)+
    (csv_load['SBP'] >= 140.0).astype(int)+
    (csv_load['DBP'] >= 90.0).astype(int)
)
q_total=info_gen+q_detail
csv_load['total_result']=(q_total >= 3).astype(int) #col 생성
#2. 파이프라인 활용
X_reg=csv_load[['Gender','Age','GLU','HbA','BMI','SBP','DBP']]
y_reg=csv_load['total_result']
#일단 자료 나눠보자
num_col=['Age','GLU','HbA','BMI','SBP','DBP']
stad_pp=Pipeline([
    ('stad',StandardScaler())
])  
merging_col_m=ColumnTransformer(
    transformers=[
        ('merging',stad_pp,num_col)
    ],
    remainder='passthrough' #Age pass
)
#3. 학습 및 튜닝
X_train,X_test,y_train,y_test=train_test_split(X_reg,y_reg,test_size=0.2,random_state=42)
doctor=Pipeline([
    ('d_stad',StandardScaler()),
    #평가 지표 설정: 로그 손실 (정답과 반대되는 오진을 남겼을 경우 로그의 성질을 이용하여 패널티를 값과 비례하여 부여(무한대로 부여 가능))
    ('xgb',XGBClassifier(random_state=42,eval_metric='logloss'))
])
#검증 데이터는 스케일링을 거치지 않음
st=StandardScaler()
X_train=st.fit_transform(X_train) #검증 데이터 때문에 한 번 더 진행 
X_test=st.transform(X_test) #스케일링 받기)
param_grid={
    'xgb__n_estimators':[100,300,500,1000],
    'xgb__max_depth':[3,5,7,11,20],
    'xgb__learning_rate':[0.1,0.01,0.001,0.000001]
}
#roc_auc
from sklearn.metrics import roc_auc_score,roc_curve

model_tuning=GridSearchCV(
    estimator=doctor,
    param_grid=param_grid,
    #roc_auc: 상세 설명은 아래
    scoring='roc_auc', 
    cv=6,
    n_jobs=-1
)
model_tuning.fit(X_train,y_train) #optimal doctor
best_model=model_tuning.best_estimator_ #정보 받은 추정자 인원 중 최적 인원 수 출력
best_scoring=model_tuning.best_score_ 
pred_data=best_model.predict(X_test) #여러 번 학습을 받은 모델
#점수 측정 방법 1
# scoring=best_model.score(X_test,y_test) #테스트 데이터로 측정
#점수 측정 방법 2
scoring=(y_test,pred_data) #테스트 데이터와 예측 데이터로 측정
final_rmse=np.sqrt(scoring) #제곱근 씌우기
print('결과 값이 작을 수록 오차 범위가 없는 거기에 좋음')
print(f'제곱근이 씌워진 평균 제곱 오차 결과: {final_rmse*100:.2f}')
#4. 시각화 자료 출력

#혼동 행렬은 '분류 지표'에 해당됨. 따라서 astype(int)를 통해 정수화 시켜서 혼동 행렬을 통해 결과 출력
#혼동 행렬 조건: 이진 데이터 취급(분류 데이터) | scatterplot: 측정 데이터 (회귀 모델과 함께)
pred_class=(pred_data>0.5).astype(int)
#array(쓰면 원본, 배열 수정본 생성)를 써서 비교 가능 (asarray는 대용량 데이터일 때만 쓰기)

#혼동 행렬(y_test,예측 데이터)
cm=confusion_matrix(y_test,pred_class)
plt.figure(figsize=(10,6))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues') #xtick,ytick 보고 결정 
plt.title('혼동 행렬을 이용한 heatmap 결과')
plt.xlabel('인공지능 예측 수치')
plt.ylabel('사실 정확도')
plt.show()
#출력 구조 2*2: 전체 코드 구조는 이진 분류로 되어있기에 0,1 두개로 나뉨

############################################################################################################################################
#ROC AUC 설명

#True,False,Positive,Negative

#Receiver Operating Characteristic: 수신자 조작 특성, 이진 데이터에 쓰이며 가장 정확한 마스터 지표
#진짜 범인(민감도,특이도)을 잡아내는 비율 vs 일반인(FPR)을 범인으로 오해하는 비율
#민감도: TP/(TP+FN), 양성(범인)을 진짜 양성(범인)으로 예측한 올바른 경우
#특이도: TN/(TN+FP), 음성(일반인)을 진짜 음성(일반인)으로 예측한 올바른 경우
#FPR(False Positive Rate)거짓 양성 비율,(1-특이도):음성(일반인)을 양성(범인)으로 예측한 잘못된 경우

#ROC Curve|
#X:FPR, 얼마나 많이 음성(일반인)을 양성(범인)으로 오해했나 <낮으면 좋음> | Y:민감도, 얼마나 많이 양성(범인)을 양성(범인)으로 맞췄나 <높으면 좋음>
#임계값(threshold)을 낮추면 FPR값이 오르는 동시에 민감도값도 같이 오름, 오해할 확률이 적어지지만, 범인을 찾지를 못함
#임계값을 올리면 민감도값이 오르는 동시에 FPR값이 오름, 범인을 잘 걸러내지만, 일반인도 거를 수 있음


#Area Under the Curve: 곡선 아래 면적: ROC그래프가 그려졌을 때, 아래의 면적을 계산한 값(1.0=만점)
#AUC 출력범위: [0.5:쓰레기, 0.7~8:쓸만함, 0.9: 구분이 확실히 가능, 1.0: 완벽]

#26.7.27