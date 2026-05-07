#본 EMR데이터는 XX대학교  극히 일부분 '틀'(내용X)을 가지고 진행했음을 알립니다
# 절대 실제 환자의 기록을 토대로 학습을 하지 않았으며 이를 가지고 유포하거나 타인에게 공유할 시, 국내 의료법에 위반된다는 걸 인지하고 있습니다.
#이번 예제는 기존 국제 의료협회에서 정해놓은 기준을 과연 모델이 군집화(정답지)를 통해 비슷하게 그룹을 형성할 수 있는 지를 테스트함(성능 테스트) 
#%%
import os
import numpy as np 
import pyreadstat
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split,GridSearchCV
#sav,csv 
sav_file='training.sav'
csv_file='csv/[csv]예제23.csv'
#convert
try:
    if not os.path.exists(csv_file):
        print('Converting...')
        sav_load,_=pyreadstat.read_sav(sav_file)
        sav_load.to_csv(csv_file,index=False)
        print('Converted!')
except Exception as e:
    print(f'ERROR: {e}')
print('-'*50)
#Explanation(metadataonly)
try:
    _,meta=pyreadstat.read_sav(sav_file,metadataonly=True)
    for name,content in meta.column_names_to_labels.items():
        label=content if content else 'None'
        print(f'{name} : {label}')
    print('loaded')
except Exception as e:
    print(f'ERROR: {e}')
#csv
csv_load=pd.read_csv(csv_file)
# #Y(정답지)
# gen_content=[
#     (csv_load['Gender']==1)&(csv_load['WC']>=90),
#     (csv_load['Gender']==2)&(csv_load['WC']>=85)
# ]
# gen_result=np.select(gen_content,[1,1],default=0)
# gen_total=(
#     (csv_load['SBP']>=120.0).astype(int)+
#     (csv_load['DBP']>=80.0).astype(int)+
#     (csv_load['GLU']>=126.0).astype(int)+
#     (csv_load['HbA']>=6.5).astype(int)
# )
# gen_total_value=gen_result+gen_total
# csv_load['positive_result']=(gen_total_value>=3).astype(int)
#X,doctor(Cluster)
X_clu=csv_load[['SBP','DBP','GLU','HbA']]
doctor_clu=Pipeline([
    ('simp',SimpleImputer(strategy='median')),
    ('stadand',StandardScaler()),
    ('km',KMeans(n_clusters=3,random_state=42))
])
#cluster column
csv_load['Cluster_col']=doctor_clu.fit_predict(X_clu)
print('-'*50)
print('Clustering Result')
group_mean=csv_load.groupby('Cluster_col')[['SBP','DBP','GLU','HbA']].mean()
for clu_num,clu_content in group_mean.iterrows():
    result_clu=f'{clu_num}:Group Number|SBP:{clu_content['SBP']:.1f}|DBP{clu_content['DBP']:.1f}|GLU:{clu_content['GLU']:.1f}|HbA:{clu_content['HbA']:.1f}'
    print(f'{result_clu}')
print('loaded')
clu_name={
    0:'Danger',
    1:'Healthy',
    2:'High Risk'
}
clu_color={
    'Danger':'orange',
    'Healthy':'green',
    'High Risk':'red'
}
csv_load['Cluster_re']=csv_load['Cluster_col'].map(clu_name)
def showgraph():
    sns.scatterplot(x='SBP',y='GLU',hue='Cluster_re',data=csv_load,style='Cluster_re',palette=clu_color) #palette:사용자 커스텀 셋 맞춰주기
    plt.legend(title='New Groups')
    plt.show()
#XGBoost
X_gb=csv_load[['Gender','Age','HT','WT','BMI','WC']]
Y_gb=csv_load['Cluster_col']
X_train_gb,X_test_gb,Y_train_gb,Y_test_gb=train_test_split(X_gb,Y_gb,test_size=0.2,random_state=42)
gen_divide=['Gender']
other_divide=['Age','HT','WT','BMI','WC']
#preprocessing
gen_col=Pipeline([
    ('gen_simp',SimpleImputer(strategy='most_frequent'))
])
other_col=Pipeline([
    ('other_simp',SimpleImputer(strategy='median')),
    ('other_stad',StandardScaler())
])
total_col=ColumnTransformer(
    transformers=[
        ('gen_total',gen_col,gen_divide),
        ('other_total',other_col,other_divide)
    ]
)
#doctor
doctor_xgb=Pipeline([
    ('main_col',total_col),
    ('xgb',XGBClassifier(random_state=42,eval_metric='mlogloss'))
    #'m'logloss:다중 분류용 오차 측정 방식,이진 분류 방식이였던 logloss에서 m(multi-class)이 추가되,AI가 각 항목에 부여한 점수를 가지고 진짜 점수와 비교
    #비교한 후 엉뚱한 정답에 확신의 비중을 진짜 정답보다 더 많이 부여했을 경우 그만큼의 손실값 적용
])
#GridSearchCV
print('Auto optimal parameter tuning started..')
param_xgb={
    'xgb__n_estimators':[50,100,150,200,300],
    'xgb__max_depth':[3,6,10],
    'xgb__learning_rate':[0.1,0.01,0.001], #오버슈팅 방지겸 0에 수렴할 수 있도록 미세조정 값 부여 
    'xgb__subsample':[0.8,1.0] #컨닝 방지용(0~1사이)
}
tuning_xgb=GridSearchCV(
    estimator=doctor_xgb,
    param_grid=param_xgb,
    scoring='accuracy',
    cv=7,
    n_jobs=-1
)
tuning_xgb.fit(X_train_gb,Y_train_gb)#학습
optimal_model=tuning_xgb.best_estimator_
best_score=tuning_xgb.best_score_ #교차검증 후 최고의 점수 출력
test_score=tuning_xgb.score(X_test_gb,Y_test_gb) #실제 모델 성능
print('-'*50)
print('Model accuracy result')
print(f'Best Cross Validate score: {best_score*100:.1f}')
print(f'Model Accuracy: {test_score*100:.1f}') 