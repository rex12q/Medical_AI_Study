#본 EMR데이터는 XX대학교  극히 일부분 '틀'(내용X)을 가지고 진행했음을 알립니다
# 절대 실제 환자의 기록을 토대로 학습을 하지 않았으며 이를 가지고 유포하거나 타인에게 공유할 시, 국내 의료법에 위반된다는 걸 인지하고 있습니다.
#%%
import pandas as pd
import numpy as np
import pyreadstat as pt
import os
import matplotlib.pyplot as plt 
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
#StratifiedKFold: 전체 데이터를 k개의 조각(fold)으로 나눈 뒤, 모든 조각이 한 번씩 검증 데이터 역할을 하도록 k번 평가를 반복하는 방식
#train_test와 비교를 하면 딱 나누어진 학습,테스트 데이터가 Random Seed에 좌우될 수 있음
from sklearn.model_selection import StratifiedKFold
#average_precision_score: 평균 정밀도 점수, 데이터가 음|양성으로 쏠려 있는 경우, (타겟에 포함된 진짜/AI가 판단한 타겟에 포함된 결과)로 나타낼 수 있다
#데이터 불균형이 심한 경우, '해당 타겟'만 기입하고 AI가 판단한 타겟을 진짜 타겟으로 나누어서 확률을 출력 (진짜만 추리는 돋보기 역할) 
from sklearn.metrics import (
    roc_auc_score, roc_curve,classification_report,
    confusion_matrix,average_precision_score
                        )

#글꼴
plt.rcParams['font.family']='AppleGothic'
plt.rcParams['axes.unicode_minus']=False

#csv -> sav
sav_file='Spss_sav/sample.sav'
csv_file='csv/claude_sample.csv' 

# try:
#     if not os.path.exists(csv_file):
#         sav_load,_=pt.read_sav(sav_file)
#         print('Converting')
#         sav_load.to_csv(csv_file,index=False)
#         print('Converted')
# except Exception as e:
#     print(f'{e}')

#csv
csv_load=pd.read_csv(csv_file)

#metadataonly
print('MetaData')
try:
    _,meta=pt.read_sav(sav_file,metadataonly=True)
    for c_name,c_label in meta.column_names_to_labels.items():
        label = c_name if c_label else 'None'
        print(f'{label} : {c_label}')
except Exception as e:
    print(f'{e}')

#Data Cleaning
#원본 sav파일 데이터 칼럼 HP,DSP 내에 '8' 존재 -> 양성(1)아닌 모름,무응답 표시
#ALG(알레르기비염) 870명 중 773명이 '8'(모름,무응답) | 모델은 (0 < 1 < 8)로 읽기에 심각한 양성으로 판단
#WT:'허리둘레A'->'체중' | BMI 공식으로 알아보기
#각 컬럼 항목에 데이터가 어떻게 분포되었는 지 사전 형태로 출력 및 확인
print('[Before] Check Col Value')
for c in ['HP','DSP','ALG']:
    #value_counts를 이용하여 칼럼 내 포함된 값 확인, to_dict로 사전 형태로 출력
    print(f'{c} : {csv_load[c].value_counts().to_dict()}')

#NaN으로 대체
#지정한 칼럼에는 '8'이라는 무응답이 있기에 NaN으로 대체
csv_load[['HP','DSP','ALG']] = csv_load[['HP','DSP','ALG']].replace(8,np.nan)

#WT = 체중 <- 확인 
e_bmi=csv_load['WT']/((csv_load['HT']/100)**2)
print(f'WT는 "체중" 컬럼이 맞는가? {(e_bmi-csv_load['BMI']).abs().max():.6f}')
print('계산된 값과 원본 데이터 내 BMI 값을 빼므로, 값이 0에 가까우면 WT = 체중')

#Data Leakage 
#조건 부여 = 조건 외우기 (앞서 짰던 예제 코드는 조건을 주어주고 같은 컬럼을 써서 데이터 누유를 방지 못 함)
#X,y
X_cla=csv_load[['Gender', 'Age', 'HT', 'WT', 'BMI', 'WC', 'SBP', 'DBP']]
#원본 데이터 손상 방지를 위해 변수 추가 후 copy(), 카피본 만들었으니 과정 생략  
y_cla=csv_load['DM'].astype(int) #당뇨병 판정 여부 (실수->정수)

#X,y sav파일 양,음성 결과 리포트
print('-'*50)
print(f'X: {X_cla}')
print('y(당뇨 판정)')
print(f'정상: {(y_cla == 0).sum()}명 / {(y_cla == 1).sum()}명')
print('-'*50)

#StratifiedKFold
#train 차이점은 아래 작성
skf=StratifiedKFold(
    n_splits=5, # 5조각으로 나누기
    shuffle=True, #섞기 허용
    random_state=42
)

#불균형 보정값: 음성/양성 비율, 현재 sav파일에는 음성: 768, 양성 102 존재, 균형잡힌 가중치가 부여되기 위하여 값을 나누기
neg, pos = (y_cla==0).sum(), (y_cla==1).sum()
unb_rate= neg/pos
print(f'양성 놓칠 시 음성 {unb_rate:.2f}명 틀린 만큼의 벌점 부여')

#CrossValidataion (스케일링 Fold 안에서 하기)
#X피처 0으로 전부 채우기 (S_KFold에 의해 전부 채워지기)
zero_prob=np.zeros(len(X_cla)) # fold1: [0,0,...,0]

#auc 정보 담기
fold_auc=[]

print('교차검증 시작 (KFold: 5)')
#enumerate: 숫자 출력, start=1과 연관지으면 터미널 출력 시 0부터 출력되는 표시를 1로 표현하여 혼동 방지
for fold,(train_idx, test_idx) in enumerate(skf.split(X_cla,y_cla),start=1):
    #cut (i(ndex)loc(ation): idx 째로 가져오기)
    X_train,X_test = X_cla.iloc[train_idx], X_cla.iloc[test_idx]
    y_train, y_test = y_cla.iloc[train_idx], y_cla.iloc[test_idx]
    #StandardScaler (fold 안에서 진행)
    st=StandardScaler()
    X_train=st.fit_transform(X_train) #fit하고 변환 
    X_test=st.transform(X_test) #fit 할 시 Leakage! 
    
    doctor= Pipeline(
        [
            #파이프라인 이용하여 별칭 부여
            ('xgbc',XGBClassifier(
        random_state=42,
        #오진일 경우 오진값 비례, 로그 성질을 이용하여 패널티 부여
        eval_metric='logloss',
        #상황에 맞게 가중치 부여 조정
        #양성 가중치 부여값: 7.53 (768/102)
        scale_pos_weight=unb_rate,
        #전체에서 80만 보고 학습(컨닝 방지)
        subsample=0.8,
        n_jobs=-1)
        )
        ]
        )

    #GridSearchCV
    xgb_grid={
        'xgbc__n_estimators':[100,200,500],
        'xgbc__learning_rate':[0.001,0.05,0.1],
        #colsample_bytree: tree당 피처에 대한 의존도 설정
        'xgbc__colsample_bytree':[0.6,0.7,1.0],
        'xgbc__max_depth':[2,3,5]
    }

    #CrossValidation 과정 층화KFold로 5번 학습
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

    #경우의 수가 100이상일 경우, RandomizedSearchCV를 씀
    tuning_doctor=GridSearchCV(
        estimator=doctor,
        param_grid=xgb_grid,
        #불균형 데이터 다룰 때 가장 '평균_정밀도' 사용
        #AUC면적 축소, 정밀도 값 떨어짐 방지
        scoring='average_precision',
        cv=cv, #층화KFold로 교차검증 5번 진행
        n_jobs=-1,
        #best_estimator 최적의 값으로 refit
        refit=True,
        #훈련용 데이터만 외우고 임하는 경우(overfitting 진단용)
        return_train_score=True
    )
    tuning_doctor.fit(X_train_st,y_train)

    #Predict_proba
    #양성 예측 확률 (refit 적용됨)
    best_prob=tuning_doctor.best_estimator_.predict_proba(X_test_st)[:,1]
    #best_prob값에 맞춰서 test_idx 부여
    zero_prob[test_idx]=best_prob
    #auc 점수 fold_auc에 리스트로 담기
    auc=roc_auc_score(y_test,best_prob)
    fold_auc.append(auc)
    print(f'FOLD {fold} | 학습 {len(train_idx)}명 / 테스트 {len(test_idx)}명'
        f'| 테스트 양성 데이터 {y_test.sum()}명 ({y_test.mean()*100:.2f}% | AUC: {auc:.4f})')

#fold별 AUC결과
#fold에 기입된 AUC결과가 결국 fold별 AUC결과로 볼 수 있음
print('-'*70)
print(f'Fold별 평균 AUC 데이터: {np.mean(fold_auc):.4f} | 표준편차: ±{np.std(fold_auc):.4f}')
print('표준편차가 작을 수록 어떤 데이터를 학습해도 실력이 일정함')
print(f'Fold 최고 기록값: {np.max(fold_auc):.4f} | Fold 최소 기록 값: {np.min(fold_auc):.4f}')


#KFold vs 층화
#KFold: 무작위 분활(비율로 표시됨)이 진행된 후, K개의 Fold(칸)으로 나눈 후, 모든 Fold에 무작위 분할 비율값이 삽입-> 삽입된 모든 fold가 검증 데이터 역할 진행
#검증된 데이터 역할 진행과 동시에 k번 반복하는 방식 (k개 fold 분할 비율 데이터값 기입-> 검증 데이터 역할과 동시에 k번 반복)

#StratifiedKFold: 층화된 KFold, 원본 데이터의 비율을 그대로 복사 후, 각 Fold에 비율값을 동일하게 삽입
#이로 인해 불균형이 심한 의료 외 다른 데이터에서도 정답이 몰리는 치명적인 오류를 차단
#k번 반복-> 1회차에 포함된 fold 칸들을 통해 경우의 수 도출 -> k회차까지 반복 (if k=5, 5번까지 반복) 
#5회차까지 도출된 출력값들을 가지고 경우의 수 병합 -> 검증 편향 방지 및 모델의 객관적인 실력을 평가할 수 있음 (균형 데이터에서 가능)
#train_test_split은 한 번 데이터 비율을 무작위로 나눈 후, Random Seed로 도출되는 방식이여서 방대하고 균등한 데이터셋일 때 사용 가능

#26.8.12
# %%
