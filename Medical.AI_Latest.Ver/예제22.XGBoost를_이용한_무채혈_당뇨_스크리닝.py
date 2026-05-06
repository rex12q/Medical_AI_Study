#본 EMR데이터는 XX대학교  극히 일부분 '틀'(내용X)을 가지고 진행했음을 알립니다
# 절대 실제 환자의 기록을 토대로 학습을 하지 않았으며 이를 가지고 유포하거나 타인에게 공유할 시, 국내 의료법에 위반된다는 걸 인지하고 있습니다.
#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyreadstat 
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#XGBoost: 잔차(Residual)학습과 경사 하강법(Gradient Decscent)를 이용해 오차가 0에 수렴하도록 활동하는 정확성 높은 모델
# 잔차 학습: 오차를 최소화하는 학습을 뜻하며 학습할 1번 Forest는 학습을 해 오차를 최소화하고 여기서 나온 찌거기 오차를 2번 Forest가 받아 오차를 최소화하는 학습을 하며 오차가 0에 수렴할 수 있게 최적의 값이 도출
# 경사 하강법: 오차를 최소화하는 과정은 경사(기울기)를 이용하여 방향을 조정하고 하강(0으로 수렴)을 통해 점점 오차가 없도록 하게 만든다. 이러한 과정에서 어떤 길을 통해 내랴가며 0에 수렴할 수 있는가를 도와주는 작업
#하지만 학습률이 너무 높은 경우 0에 수렴하긴 커녕 다시 올라가서 오차와의 수렴값이 멀어질 수도 있어서 학습을 안전하게 고쳐나간다(learning_rate)
#멀티스레딩(동시에 여러 작업 수행)을 사용하기에 OpenMP라는 도구를 사용함(brew있을 시, brew install libomp)
sav_file = 'training.sav'
csv_file = 'csv/[csv]XGBoost_test.csv'
try:
    if not os.path.exists(csv_file):
        print('Converting..')
        sav_load,_=pyreadstat.read_sav(sav_file)
        sav_load.to_csv(csv_file,index=False,encoding='utf-8-sig')
        print('Converted')
except Exception as e:
    print(f'ERROR: {e}')
print('-'*50)
#metadataonly
try:
    _,meta=pyreadstat.read_sav(sav_file,metadataonly=True)
    for c_name,c_label in meta.column_names_to_labels.items():
        label=c_label if c_label else 'Not Explanation'
        print(f'Information: {label}')
    print('loaded')
except Exception as e:
    print(f'ERROR: {e}')
print('-'*50)
#csv_load
csv_load = pd.read_csv(csv_file)
#Hidden_Diabetes
#np.where:(조건,참일 때,거짓일 때),조건이 하나만 있을 때,elect처럼 조건의 갯수를 따지면 리스트를 안 써도 됨
csv_load['Hidden_Diabetes']=np.where(csv_load['HbA']>=6.5,1,0)
#data
X_gb=csv_load[['Gender','Age','HT','WT','BMI','WC','SBP','DBP']]
Y_gb=csv_load['Hidden_Diabetes'] #환자의 겉모습만 보고 진짜 당뇨를 찾아내기(XGBoost 성능 테스트)
#train_test
X_train,X_test,Y_train,Y_test = train_test_split(X_gb,Y_gb,test_size=0.2,random_state=42)
freq_part=['Gender']
median_part=['Age','HT','WT','BMI','WC','SBP','DBP']
#preprocessing
freq_pp=Pipeline([
    ('freq',SimpleImputer(strategy='most_frequent'))
])
median_pp=Pipeline([
    ('median',SimpleImputer(strategy='median')),
    ('stad',StandardScaler())
])
main_trans=ColumnTransformer( #순서:(이름,기계,컬럼명)
    transformers=[
        ('med_pp',median_pp,median_part),
        ('fr_pp',freq_pp,freq_part)
    ]
)
#doctor
doctor_xgb=Pipeline([
    ('main',main_trans), #결측치 채우고 스케일링
    ('xgb',XGBClassifier(random_state=42,eval_metric='logloss'))
    #use_label_encoder:사용자의 파일이 숫자가 아닌 문자로 되어있을 경우, 숫자로 엔코딩하는 작업 (그러나 이 작업은 개발자들의 항의로 없어졌다고 한다)
    #eval_metric(평가지표)logloss:어느 정도의 오차가 조금이라도 났을 경우 상황에 맞게 오차에 비례한 손실값을 부여함(교차 엔트로피)
    #모델이 예측한 확률 분포와 실제 정답의 확률 분포 사이의 차이를 측정
])
#GridSearchCV
main_tuning={
    'xgb__n_estimators':[100,200,300], #오답 체크(릴레이 방식)를 몇 번이나 할 것인가?
    'xgb__max_depth':[3,5,7], #나무(의사) 1개 당 얼마나 깊이 생각을 할 것인가?
    'xgb__learning_rate':[0.01,0.1], #미세조정을 해서 0에 수렴하도록 도와주는 역할(오버슈팅이 일어날 경우를 대비해 값을 크게 부여 안 함)
    'xgb__subsample':[0.8,1.0] #subsample: 데이터 전체를 외우는 것을 방지하기 위해 일부만 가리게 하는 기능
}
print('XGBoost Hidden_Diabetes hunter started..')
main_grid=GridSearchCV(
    estimator=doctor_xgb,
    param_grid=main_tuning,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)
main_grid.fit(X_train,Y_train)
best_model=main_grid.best_estimator_
best_scoring=main_grid.best_score_
test_score=best_model.score(X_test,Y_test) #베스트 모델로 성능 테스트
print('-'*50)
print('Result')
for model_name,model_info in main_grid.best_params_.items():
    print(f'{model_name} : {model_info}')
print('-'*50)
print(f'Train (CV) accuracy: {best_scoring*100:.1f}%') #교차검증 중 최고의 퍼포먼스를 보여준 점수 출력
print(f'Final test accuracy: {test_score*100:.1f}%') #최고의 퍼포먼스를 가진 모델이 직접 실전 문제를 풀어 출력
print('-'*50)
print('Result Graph')
fig,axes=plt.subplots(1,2,figsize=(14,6)) # 화면 분활, 꽉 채우기
sns.countplot(x='Hidden_Diabetes',data=csv_load,ax=axes[0])
axes[0].set_title("XGBoost's factor")
axes[0].set_xticks([0,1],['Negative','Positive'])
sns.scatterplot(x='Age',y='HbA',data=csv_load,hue='Hidden_Diabetes',alpha=0.4,ax=axes[1])
axes[1].axhline(6.5,color='red',linestyle='--') #6.5이상은 양성이라고 수평선 추가해주기
axes[1].set_title('Hemoglobin Scatterplot')
axes[1].legend(title='Diabetes',labels=['Negative(0),Positive(1)'])
plt.tight_layout()
plt.show()
# %%