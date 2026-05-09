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
from sklearn.metrics import confusion_matrix
#confusion_matrix: 오차 행렬을 나타낼 수 있는 사이킷런.지표 도구 중 하나
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
# def showgraph():
sns.scatterplot(x='SBP',y='GLU',hue='Cluster_re',data=csv_load,style='Cluster_re',palette=clu_color,ax=axes[0]) #palette:사용자 커스텀 셋 맞춰주기
axes[0].legend(title='New Groups')
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
#Confusion Matrix
print('Drawing confusion_matrix...')
y_pred=tuning_xgb.predict(X_test_gb)#성능 테스트를 위한 실제 문제 풀기
confusion_test=confusion_matrix(Y_test_gb,y_pred)
con_names=['Danger','Healthy','High Risk'] #0,1,2이름 맞춰 쓰기(3*3행렬표)
sns.heatmap(
    confusion_test,annot=True,fmt='d',cmap='Blues',ax=axes[1],
    xticklabels=con_names,yticklabels=con_names #각 칸마다 이름 부여(ticklabels)
    )
#heatmap:데이티 사이의 상관관계나 수치 강도를 색상으로 시각화할 때 쓰는 도구(얼마나 연관성이 있는 지를 색을 통해 제공하는 시각적 도구)
#annot:칸 안에 숫자를 기입하고 싶을 때|fmt(format:형식)='d'(Decimal Integer:10진수 정수),소수점까지 나타내고 싶으면 '.1f'
axes[1].set_xlabel('Predicted My AI Model') #인공지능이 예측한 것
axes[1].set_ylabel('Actual (y_pred)') #실제 정답
axes[1].set_title('Confusion Matrix Result')
plt.tight_layout()
plt.show()
print('End!')
#오차 행렬표 의미와 해석
#색의 진함:93이라는 수치를 띄고 있는 칸은 사람이 가장 몰렸다는 뜻
#정답 대각선(왼쪽 위에서 오른쪽 아래):정답 대각선을 기준으로 해서 보면 색을 가장 진하게 띄고 있는 93, 정확도가 가장 높다는 뜻(참고로 칸 하나하나는 그저 사람이 많다는 의미로 해석)
#나머지 칸(오답 구역):대각선을 벗어난 칸들의 색이 짙어진다면, AI가 답을 많이 틀렸다는 뜻(심각한 문제)
#1번째 줄(Danger:중위험):중위험 단계에선 [Danger:0,Healthy:9,High Risk:1] 결과가 나왔으며, 모델한테 어느 정도의 손실점수를 부여->모델 성능 손실
#2번째 줄(Healthy:건강):건강 단계에선 [Danger:0,Healthy:93,High Risk:9] 결과가 나왔으며, 모델한테 많은 향상점수를 부여->모델 성능 향상
#3번째 줄(High Risk:고위험):고위험 단계에선 [Danger:0,Healthy:49,High Risk:13] 결과가 나왔으며
#고위험 단계에서 사람들을 건강이라고 판정한 모델의 성능이 여기서 심각하게 저하된 걸 알 수 있음(고위험에서 43명한테 정상판정을 주었으니..)
#결론
#AI는 겉모습 데이터만으로는 숨겨진 병을 찾지 못하며 AI 스크리닝의 한계와 혈액 검사의 필수성이 아주 강력히 필요하다는 것을 알 수 있음