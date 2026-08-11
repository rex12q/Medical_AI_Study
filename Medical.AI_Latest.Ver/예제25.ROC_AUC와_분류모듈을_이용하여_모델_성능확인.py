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
#조건과 같은 컬럼을 쓸 시 데이터 유출로 판단 
#당뇨병 여부, 거주 지역, 교육 수준 참고
X_cla=csv_load[['Gender','Age','DM','Education','HP','DSP']]
y_cla=csv_load['total_result']
#일단 자료 나눠보자
num_col=['Age','GLU','HbA','BMI','SBP','DBP']
stad_pp=Pipeline([
    ('stad',StandardScaler())
])  
merging_col_m=ColumnTransformer(
    transformers=[
        ('merging',stad_pp,num_col)
    ],
    remainder='passthrough' #범주형 데이터 패스
)
#3. 학습 및 튜닝
X_train,X_test,y_train,y_test=train_test_split(X_cla,y_cla,test_size=0.2,random_state=42)
doctor=Pipeline([
    ('d_stad',StandardScaler()),
    #평가 지표 설정: 로그 손실 (정답과 반대되는 오진을 남겼을 경우 로그의 성질을 이용하여 패널티를 값과 비례하여 부여(무한대로 부여 가능))
    ('xgb',XGBClassifier(random_state=42,eval_metric='logloss'))
])
#검증 데이터는 스케일링을 거치지 않음
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

#roc_curve 생성을 위해 predict_proba로 설정 [전체,양성값] (4번 과정으로 이어짐)
prob_data=best_model.predict_proba(X_test)[:,1] #여러 번 학습을 받은 모델
pred_data=best_model.predict(X_test)

#점수 측정 방법 (최고의 임계값을 찾기 전)
acc=accuracy_score(y_test,pred_data) #테스트 데이터와 예측 데이터로 채점
f1=f1_score(y_test,pred_data)
c_report=classification_report(y_test,pred_data)
r_score=roc_auc_score(y_test,prob_data)
print()
print('최고의 임계값을 찾기 전, 모델 성능 평가')
print()
print(f'모델의 정확성(Accuracy_Score): {acc*100:.2f}점 ')
print()
print('분류 리포트')
print(f'{c_report}')
print()
print(f'정확성 뒷받침 근거 채점 자료(f1_score): {f1*100:.2f}% ')
#생사람 안 잡고, 양성을 몇 명 발견했나 [분모: 모델이 양성이라고 찍은 전체] (정밀도), 실제 양성 데이터에서 모델이 찾아낸 비율 [분모: 진짜 양성 데이터 전체] (재현율)
print('정확성 지표에서 자료를 재현율과 정밀도를 이용하여 보다 정확한 지표 출력')
print()
print(f'ROC_AUC_SCORE: {r_score*100:.2f}점')
print('록_스코어는 최적의 임계값을 구하기 전, 모델이 모든 경우의 임계값을 통해 테스트한 전체 결과를 지표임을 알림')
print()

#4. 시각화 자료 출력

#FPR,TPR을 통해 최적의 임계값 찾기| 내부 순서 규칙:(오답,정답,임계값)
fpr,tpr,thresholds=roc_curve(y_test,prob_data)
#순이익 계산: tpr(수익)-fpr(비용)=순이익
optimal_value=np.argmax(tpr-fpr) #np.argmax:가장 큰 값이 존재하는 idx 불러오기
#최고 임계값은 데이터가 변형될 시 언제든 변동 가능성 유 
best_value= thresholds[optimal_value] #리스트로 표현
#직접 확인
print(f'최고의 임계값 범위: {best_value:.2f}')

#정해진 최소 양성 범위 0.5 조건 삭제 -> 최적의 임계값으로 직접 조건 설계(조건 넘으면 1)
prob_class=(prob_data>best_value).astype(int)

#최고 임계값을 찾은 후 지표 출력
new_acc=accuracy_score(y_test,prob_class)
new_report=classification_report(y_test,prob_class)
new_f1=f1_score(y_test,prob_class)
print()
print('최고의 임계값을 찾은 후, 모델 성능 평가')
print()
print(f'모델의 정확성(Accuracy_Score): {new_acc*100:.2f}점 ')
print()
print('분류 리포트')
print(f'{new_report}')
print()
print(f'정확성 뒷받침 근거 채점 자료(f1_score): {new_f1*100:.2f}% ')
#양성을 양성으로 판단(정밀도),올바르게 판단한 후, 양성을 몇 명 발견했나(재현율)
print('정확성 지표에서 자료를 재현율과 정밀도를 이용하여 보다 정확한 지표 출력')


#roc_curve(y_데이터,prob_data)
plt.figure(figsize=(10,6))
#록_커브가 적용된 시각화 자료
plt.plot(fpr,tpr,color='blue',label='ROC Curve (AI)')
#예시용(AUC: 0.5 그래프) 대각선 그래프: 록_커브가 적용된 그래프가 예시용 그래프와 비교했을 때 얼마나 튀는가?
plt.plot([0,1],[0,1],color='red',linestyle='--',label='Example(AUC:0.5)')
#최적의 임계값 fpr,tpr을 통해 별로 표시
plt.scatter(fpr[optimal_value],tpr[optimal_value],color='green',marker='*',label=f'최고 임계값: {best_value:.2f}')
plt.title('록 커브, 최적의 임계값 찾기')
plt.xlabel('FPR (음성을 양성이라고 오해한 비율(전체 비율 낮으면 좋음))')
plt.ylabel('TPR (양성을 양성이라고 맞춘 비율(전체 비율 높으면 좋음))')
plt.legend() #label 모아주기
plt.grid(True) #바둑판 격자 활성화
plt.show()
#출력 구조 2*2: 전체 코드 구조는 이진 분류로 되어있기에 0,1 두개로 나뉨

############################################################################################################################################
#ROC AUC 설명

#임계값(threshold): 범위 지정 값은 소수점(0~1)으로 설정됨 

#용어: True,False,Positive,Negative

#Receiver Operating Characteristic: 수신자 조작 특성, 이진 데이터에 쓰이며 가장 정확한 마스터 지표
#진짜 범인(민감도,특이도)을 잡아내는 비율 vs 일반인(FPR)을 범인으로 오해하는 비율
#민감도: TP/(TP+FN), 양성(범인)을 진짜 양성(범인)으로 예측한 올바른 경우
#특이도: TN/(TN+FP), 음성(일반인)을 진짜 음성(일반인)으로 예측한 올바른 경우
#FPR(False Positive Rate)거짓 양성 비율,(1-특이도):음성(일반인)을 양성(범인)으로 예측한 잘못된 경우

#ROC Curve|
#X:FPR, 얼마나 많이 음성(일반인)을 양성(범인)으로 오해했나 <생사람 잡은 비율> | Y:TPR 민감도, 얼마나 많이 양성(범인)을 양성(범인)으로 맞췄나 <진실을 맞춘 비율>
#임계값(threshold)을 낮추면 양성값을 전부 잡을 수 있지만, 음성을 양성으로 오해하는 비율이 급증함
#임계값을 올리면 민감도값이 떨어지는 동시에 FPR값도 떨어짐, 양성을 양성으로 잘 잡지만, 양성을 음성으로 판단할 수 있음

#ROC Curve는 predict_proba모듈 내장어를 이용하여 생성되며, 생성된 그래프에서 Under Curve가 AUC로 잡힘

#Area Under the Curve: 곡선 아래 면적: ROC그래프가 그려졌을 때, 아래의 면적을 계산한 값(1.0=만점)
#AUC 출력범위: [0.5:쓰레기, 0.7~8:쓸만함, 0.9: 구분이 확실히 가능, 1.0: 완벽]
#####################################################################################################################################
#ROC_AUC 결과

#EMR 전체 데이터셋: 정상 151, 고위험 23

#확인하는 방법
#수직 상승: TPR Rate Up | 수평: FPR Rate Up

#ROC 파트
#초반에는 양성을 양성으로 찾는 과정이 많았지만, 중후반부로 갈수록 음성을 양성으로 오해하는 비율이 높아짐

#Thresholds 파트
#임계값 0.1 지점을 찾았을 때| TPR:0.8 | FPR: 0.4
#TPR 수치는 80%, 실제 양성 데이터 중 80%를 성공적으로 양성 데이터로 판단 
#FPR 수치 40%, 실제 음성 데이터 중 40%를 음성 데이터를 양성으로 판단 (오진)

#AUC 파트
#0.5 AUC 밑으로 안 떨어진 걸 확인할 수 있음 -> 모델 성능이 Example 대각선 아래로 추락할 시 모델의 학습 또는 데이터를 물갈이


#26.7.29
