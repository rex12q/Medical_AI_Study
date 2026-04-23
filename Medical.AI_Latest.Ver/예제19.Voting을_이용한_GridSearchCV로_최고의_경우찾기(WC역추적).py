#본 EMR데이터는 XX대학교  극히 일부분 '틀'(내용X)을 가지고 진행했음을 알립니다
# 절대 실제 환자의 기록을 토대로 학습을 하지 않았으며 이를 가지고 유포하거나 타인에게 공유할 시, 국내 의료법에 위반된다는 걸 인지하고 있습니다.
#이번 예제는 '혈액수치'를 가지고 '뱃살'(허리둘레)를 추적하는 시스템을 설계한다, ML이 과연 예측 모델로써 충분한 역할을 할 지 알아보자
#%%
import numpy as np
import pandas as pd
import os 
import pyreadstat as pt #sav 읽기
from sklearn.ensemble import VotingClassifier #SVC + RF
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC #Support Vector Machine: Support Vector Classifier
from sklearn.pipeline import Pipeline # 둘의 차이점은 직접 이름을 달아주느냐 달아주지 않느냐에 대한 자동,수동 차이 (make_pipline:자동|pipeline:수동)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
#Grid(격자)Seach(탐색)CV(Cross Validation,교차검증)
#Grid:(나무50,나무100)과 (깊이 3과 5단계)중 골라 경우의 수를 계산 -> 경우의 수를 곱해 격자(Grid)모형을 만든다
#Search: 나온 n개의 조합을 가지고 하나하나 다 학습시킨다
#Cross Vaildation: 학습해서 가장 성능이 우수한 '1'이 있을 경우, 실수가 안 나오게 계속 모델을 (학습->테스트) 해서 진짜 성능이 우수한 n을 뽑아 검증을 한다 
from sklearn.model_selection import train_test_split, GridSearchCV

sav_file = 'training.sav'
csv_file = 'csv/[csv]Blood_WC.csv' #[csv]->csv/[csv]

#sav->csv
if not os.path.exists(csv_file):
    try:
        savCH,_ = pt.read_sav(sav_file)
        print('Converting')
        savCH.to_csv(csv_file,index=False,encoding='utf-8-sig')
        print('Complete converting')
    except Exception as e:
        print(f'ERROR: {e}')
csv_load = pd.read_csv(csv_file)
#metadataonly
try:
    _,meta = pt.read_sav(sav_file,metadataonly=True)
    for colName,coLabel in meta.column_names_to_labels.items():
        c_label = coLabel if coLabel else 'Not explanation'
        print(f'Explantion: {coLabel}')
        print('Complete load')
except Exception as e:
    print(f'ERROR: {e}')

#male,female (기준이 다르기에 나눠야 함) 
condition = [
    (csv_load['Gender'] == 1) & (csv_load['WC'] > 90), #male
    (csv_load['Gender'] == 2) & (csv_load['WC'] > 85) #female
] #&:리스트(시리즈)전체 전용 pandas는 한 줄씩 개별적으로 계산할 때 &를 쓸 수 있다

csv_load['Abdominal_Obesity'] = np.select(condition, [1,1], default=0)
#np.select: 조건 선택 [n1,n2] | 만약 조건을 선택할 칸에 조건이 2개 있을 경우 조건1=n1,조건2=n2 이렇게 됨 defalut는 칸에 있는 조건이 만족 못 했을 경우 출력되는 값
#구식인 for반복문 보단 '백터화'를 통해 행렬에 조건을 전부 나타낼 수 있는 np를 많이 선호함

# condition = [] 구식...
# for c in range(len(csv_load)):
#     if csv_load['Gender'][c] == 1:
#         if csv_load['WC'][c] > 90:
#             condition.append(1)
#     elif csv_load['Gender'][c] == 2:
#         if csv_load['WC'][c] > 85:
#             condition.append(1)
#     else:
#         condition.append(0)
# csv_load['Abdominal Obesity'] = condition

#train_test
X_all = csv_load[['Age','SBP','DBP','GLU','DM']]
Y_all = csv_load['Abdominal_Obesity'] #답지

X_train,X_test,Y_train,Y_test=train_test_split(X_all,Y_all,random_state=42,test_size=0.2)

#pipeline
doctor_cla = RandomForestClassifier(random_state=42)
doctor_svc = SVC(probability=True,random_state=42)
doctor_vote = VotingClassifier(
    estimators=[
    ('cla',doctor_cla), # ':'(x), ','(o)
    ('svc',doctor_svc),
    ], 
    voting = 'soft' #두 Ml의 평균을 내줌
)
main_pp = Pipeline([ #파이프라인 따로 구축 -> GridSearchCV때 유용하게 써야 함
    ('impute',SimpleImputer(strategy='median')), #결측치값 '중앙'
    ('stad',StandardScaler()),
    ('vote',doctor_vote)
])

#GridSearchCV
param_grid = {
    #랜덤포레스트 튜닝 설정(추정자,질문 개수)
    'vote__cla__n_estimators': [50,100,150], #파이썬에서는 폴더 안으로 들어갈 때 '__'언더바 2개를 쓴다
    #vote->cla->n_estimators 추정자 조정(최적의 환경이 나올 때까지)
    'vote__cla__max_depth': [3,5,7],
    #max_depth(질문의 수): 최대 깊이가 어느 정도든 가장 최적의 환경이 나올 때까지 노가다가 필요

    #svc 튜닝 설정
    #C:cost(비용) 또는 penalty(벌점)의 약자: 선을 그을 때 약간의 오차를 얼마나 봐줄 것인가(그러므로 '정수'가 아닌 '실수'의 형태를 가짐,가중치(w))
    #(얼마나 깐깐하게 볼 것인가(한 치의 오차도 없이->값이 매우 높음)|느슨해도 괜찮으니 전체적으로 참고하기(값이 매우 낮음))
    'vote__svc__C': [0.0001,0.001,0.1,1.0,10.0,100.0] #svc는 테스트를 할 때 미세하게 조정하면 별 의미가 없기에 10배 100배 단위로 테스트
    } #Dictionary 형태 ':'

print('Auto tuning factory started..!')

#GridSearchCV 트레이닝(공장 가동,full setting)
grid_test = GridSearchCV(
    estimator=main_pp, #main_pp를 통쨰로 가지고 학습을 할 것, (모델 객체 지정)
    param_grid=param_grid, #위에서 만든 param_grid 딕셔너리로 주어진 값으로 테스트, (하이퍼파라미터 목록을 딕셔너리로 전달)
    cv=5, #한번만 했을 경우 우연히 쉬운 문제가 나와 그대로 출력이 될 수 있기 때문, (교차 검증 n번 시키기)
    scoring='accuracy', #정확성 점수와 이에 관한 정밀도로 모델의 성능을 확실히 검증하기 위해 사용 (accuracy:단순히 답을 많이 맞춘 경우,f1: 정밀도)
    n_jobs=-1#코어(일꾼들) 총동원, 만약 1이면 코어 하나, 그러나 시간 단축을 위해 모든 코어를 다 끌어다 씀(영끌)
) 
grid_test.fit(X_train,Y_train)
print('-'*50)
print('Result obesity prediction')
print('-'*50)
#사용자 이름 별도로 설정, grid_test를 베스트 선정 (best_params <- 약속된 단어)
for param_name,param_value in grid_test.best_params_.items(): #.items():가져오기
    print(f'{param_name} : {param_value}') 
best_model = grid_test.best_estimator_ #best_estimator: 실전용 완성 모델을 보여줌
test_acc = best_model.score(X_test,Y_test) #모델의 정확성을 확인하기
print(f'Model accuracy: {test_acc*100:.2f}')
print('END')
#모델이 X를 학습하고 사용자가 직접 양성 범위(Y)를 설정해 최적의 학습을 시키기 위해 GridSearchCV를 써서 최고의 경우를 찾아줬다
# %%
