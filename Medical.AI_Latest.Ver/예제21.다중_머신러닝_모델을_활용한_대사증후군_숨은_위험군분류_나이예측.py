#본 EMR데이터는 XX대학교  극히 일부분 '틀'(내용X)을 가지고 진행했음을 알립니다
# 절대 실제 환자의 기록을 토대로 학습을 하지 않았으며 이를 가지고 유포하거나 타인에게 공유할 시, 국내 의료법에 위반된다는 걸 인지하고 있습니다.
#%%
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pyreadstat
import os
from sklearn.model_selection import train_test_split,GridSearchCV #학습과 테스트,최고의 성능을 찾기
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,VotingClassifier #분류,예측,voting
from sklearn.svm import SVC #voting될 ML
from sklearn.pipeline import Pipeline,make_pipeline #이름 직접 지어주고 기능 전부 넣기
from sklearn.preprocessing import StandardScaler #표준화 스케일링(단위에 따른 값 맞추기)
from sklearn.impute import SimpleImputer #결측치 채우기
from sklearn.cluster import KMeans #비지도 학습을 위한 군집화 
#compose:서로 다른 열에서 서로 다른 전처리 단계를 적용하고 하나로 합칠 때 사용하는 도구
from sklearn.compose import ColumnTransformer #열 분배기: 결측치 설정값에 서로 다른 값을 부여하기 위해 씀
sav_file='Spss_sav/training.sav'
csv_file='csv/[csv]Metabolic_Syndrome.csv' #생성 완료
#sav->csv
try:
    if not os.path.exists(csv_file):
        sav_load,_=pyreadstat.read_sav(sav_file)
        print('Converting... sav->csv')
        sav_load.to_csv(csv_file,index=False,encoding='utf-8-sig') #정보 없는 행열에 번호 부여 방지,한글 지원
        print('Converted!')
except Exception as e:
    print(f'ERROR: {e}')
#metadatonly
print('-'*50)
print("Sav file's Metadata(Explanation)")
try:
    _,meta=pyreadstat.read_sav(sav_file,metadataonly=True)
    for savName,savCol in meta.column_names_to_labels.items():
        Col=savCol if savCol else 'Not Explanation'
        print(f'Label : {Col}')
    print('Complete load!')
except Exception as e:
    print(f'ERROR: {e}')
csv_load=pd.read_csv(csv_file)
csv_edit=csv_load.drop(columns=['ID']) #ID는 필요없음
#정답지 열 따로 만들기
other_wc=[
    (csv_edit['Gender'] == 1)&(csv_edit['WC']>=90),
    (csv_edit['Gender'] == 2)&(csv_edit['WC']>=85)
]
info_wc=np.select(other_wc,[1,1],default=0)
info_pos=( #조심...
    (csv_edit['SBP']>=140.0).astype(int)+
    (csv_edit['DBP']>=90.0).astype(int)+
    (csv_edit['BMI']>=30.0).astype(int)+
    (csv_edit['GLU']>=126.0).astype(int)+
    (csv_edit['HbA']>=6.5).astype(int)
)
total_result=info_wc+info_pos
csv_edit['positive_result']=(total_result>=3).astype(int) #정답지 열 생성! 
print('-'*50)
#KMeans
X_clu=csv_edit[['SBP','DBP','GLU','HbA','BMI','WC']]
doctor_clu=make_pipeline(
    SimpleImputer(strategy='median'), #결측치 '중앙'
    StandardScaler(),
    KMeans(n_clusters=4,random_state=42) # 0,1,2,3군집화,섞는 방법 고정
)
csv_edit['Cluster_col']=doctor_clu.fit_predict(X_clu) #비지도 학습이기에 fit_predict를 써야 함
print("EMR file's Cluster Graph")
group_clu = {
    0:'Healthy People(1)',
    1:'High-Risk Metabolic_Syndrome(4)',
    2:'Hypretension Obesity(3)',
    3:'Meatbolic Healthy Overweight(2)'
}
color_clu = {
    0:'green',
    1:'red',
    2:'pink',
    3:'blue'
}
group_mean=csv_edit.groupby('Cluster_col')[['SBP','DBP','GLU','HbA','BMI','WC']].mean() #그룹별 평균
for clu_num,clu in group_mean.iterrows():
    result_clu=f"[{clu_num}.Group]SBP:{clu['SBP']:.1f}|DBP:{clu['DBP']:.1f}|GLU:{clu['GLU']:.1f}|HbA:{clu['HbA']:.1f}|WC:{clu['WC']:.1f}]"
    print(result_clu)
csv_edit['Cluster_re']=csv_edit['Cluster_col'].map(group_clu) # 이름 설정
#scatterplot
def showgraph():
    sns.scatterplot(x='BMI',y='WC',data=csv_edit,hue='Cluster_re',style='Cluster_col')
    plt.title('Correlation Body Mass Index and Waist Circumference') 
    plt.xlabel('Body Mass Index(BMI)')
    plt.ylabel('Waist Circumference(WC)')
    text_description = (
        '''Why this plot?: BMI and WC is key indicator of decision.
        Looking at the scatter plot, we can check how body features are categorized through clustering.
        In the top of right, Abdominal obesity and body's obesity is seriously danger levels. 
        As you can see, this in the 'High-Risk'! '''
    )
    plt.figtext(0.5,0.05,text_description,wrap=True,horizontalalignment='center') 
    #figtext:설명 문구 집어넣기|가로:0.5,세로:0.05|wrap=True: 문장이 길어질 시 알아서 바꿔줌|수평:중앙
    plt.subplots_adjust(bottom=0.25) #adjust: 조절 -> bottom(아래,25프로 공간 남겨두기)
    plt.show()
print('-'*50)
print('Auto factory started...')
#train_test
X_vote=csv_edit[['Gender','HT','WT','BMI','WC']]#피 뽑기 전 스탯
#이 모델은 X와 Y의 상관관계가 매우 작아 정확도가 낮을 수도 있다. 하지만 피 뽑기 전 스탯을 이용해 어떤 그룹에 속할 지 예측을 한다는 점에서 의의가 있다.
#일부러 Y_clu의 스탯을 전부 안 겹치도록 설계함
Y_clu=csv_edit['Cluster_re']#(군집화 그룹을 Y)
X_reg=csv_edit[['Gender','HT','WT','BMI','WC','SBP','DBP','GLU','HbA']]#나이예측(위와는 별개이므로 모든 스탯 총동원)
Y_reg=csv_edit['Age']#(drop하는 과정 생략)
X_train_vote,X_test_vote,Y_train_clu,Y_test_clu=train_test_split(X_vote,Y_clu,test_size=0.2,random_state=42)
X_train_reg,X_test_reg,Y_train_reg,Y_test_reg=train_test_split(X_reg,Y_reg,test_size=0.2,random_state=42)
#doctor
doctor_rf=RandomForestClassifier(random_state=42)
doctor_svc=SVC(probability=True,random_state=42)
doctor_reg=RandomForestRegressor(random_state=42)
doctor_vote=VotingClassifier(
    estimators=[
        ('rf',doctor_rf),
        ('svc',doctor_svc)
        ],voting='soft' #이름 지어주고 모델 합치기
)
#preprocessing
#ColumnTransformer(1)
median_simp=['HT','WT','BMI','WC'] #전처리 과정에서 결측치 중앙값을 받을 항목들,(X에는 'Age'데이터가 없음)
freq_simp=['Gender'] #전처리 과정에서 결측치 최빈값을 받을 항목들
#pipeline
median_pp=Pipeline([ 
    ('med_simp',SimpleImputer(strategy='median')),
    ('sta',StandardScaler())
])
freq_pp=Pipeline([
    ('freq_simp',SimpleImputer(strategy='most_frequent'))
    ])
#ColumnTransformer(2) 규칙:(이름,기계,컬럼)
main_trans=ColumnTransformer(
    transformers=[
        ('med',median_pp,median_simp),
        ('freq',freq_pp,freq_simp)
    ])
#Final Pipeline
pipe_vote=Pipeline([
    ('main_col',main_trans),
    ('vote',doctor_vote) #전처리가 된 데이터 받기(vote)
])
pipe_reg=Pipeline([
    ('main_col',main_trans),
    ('reg',doctor_reg) #전처리가 된 데이터 받기(reg)
])
#GridSearchCV
main_vote_grid={ #dict 형태
    #rf
    'vote__rf__n_estimators':[10,50,100,200], #추정자들(나무)개수
    'vote__rf__max_depth':[2,10,50], #질문의 깊이(과소적합과 과적합을 조심해야함)
    #SVC,'C'
    'vote__svc__C':[0.1,10.0,100.0] #얼만큼의 오차를 허용할 것인가->C
}
main_reg_grid={
    #reg
    'reg__n_estimators':[10,50,100,200],
    'reg__max_depth':[2,10,50]
}
optimal_vote_grid=GridSearchCV(
    estimator=pipe_vote,
    param_grid=main_vote_grid,
    scoring='accuracy',
    cv=5, # 교차검증
    n_jobs=-1
)
optimal_reg_grid=GridSearchCV(
    estimator=pipe_reg,
    param_grid=main_reg_grid,
    #MSE는 벌점 시스템(음수->양수)이기에,단순한 그리드서치의 채점 방식에 오류가 발생->neg(음수)를 붙여서 최적의 모델 뽑기(-50<-10)
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1
)
optimal_vote_grid.fit(X_train_vote,Y_train_clu) #vote
optimal_reg_grid.fit(X_train_reg,Y_train_reg) #reg
print('Completely ended!')
print('-'*50)
print('Total Performance')
#vote
best_vote_model=optimal_vote_grid.best_estimator_
best_vote_score=optimal_vote_grid.best_score_
test_score=best_vote_model.score(X_test_vote,Y_test_clu) #실력 테스트
#reg
best_reg_model=optimal_reg_grid.best_estimator_
best_reg_score=optimal_reg_grid.best_score_
real_mse=best_reg_score * -1 #MSE특징인 음수 출력을 방지하기 위한 안전장치
sqrt_mse=np.sqrt(real_mse) #제곱한 값을 루트로 다시 벗겨주기
print('-Best voting model and score-')
print(f'Model: {best_vote_model}')
print(f'Train (Cross Validate) test: {best_vote_score*100:.2f}')
print(f'Final Score: {test_score*100:.2f}')
print('-Best voting model and score-')
print(f'Model: {best_reg_model}')
print(f'Score: {sqrt_mse:.2f}')
#UserInfo
print('-'*50)
while True:
    print("AI can check user's status!") 
    def replace_user():
        print('You can enter only (y|n)')
        #1.BP
        while True:
            s_dbpQ = input('Q1. Have you ever been diagnosed with high blood pressure, or are you taking blood pressure pills? (y/n): ')
            if s_dbpQ == 'y':
                user_sbp = 145.0
                user_dbp = 95.0
                print('Answer recorded!')
                break
            elif s_dbpQ == 'n':
                user_sbp = 120.0
                user_dbp = 80.0
                print('Answer recorded!')
                break
            else:
                print("Please enter 'y' or 'n' only.")
        #2.GLU
        while True:
            gluQ = input('Q2. Have you ever been told you have "high blood sugar" in a recent health check-up? (y/n): ')
            if gluQ == 'y':
                user_glu = 126.0
                print('Answer recorded!')
                break
            elif gluQ == 'n':
                user_glu = 95.0
                print('Answer recorded!')
                break
            else:
                print("Please enter 'y' or 'n' only.")
        #3.Exercise
        while True:
            user_q = input('Q3. Do you exercise enough to sweat at least 3 times a week? (y/n): ')
            if user_q == 'y':
                user_sbp -= 5.0
                user_dbp -= 5.0
                print('Answer recorded!')
                break
            elif user_q == 'n':
                user_sbp += 5.0
                user_dbp += 5.0
                print('Answer recorded!')
                break
            else:
                print("Please enter 'y' or 'n' only.")
        return user_sbp,user_dbp,user_glu
    def userRange(prompt,minval,maxval,is_float=True):
        while True:
            try:
                uRange=float(input(prompt)) if is_float else int(input(prompt))
                if minval <= uRange <= maxval:
                    return uRange
                print('Over range, check your value!')
            except Exception as e:
                print(f'ERROR: {e}')
    u_gender=[userRange('Enter your gender number (1:Male,2:female): ',1,2,is_float=False)]
    u_age=[userRange('Enter your age (Range:0~150): ',0,150,is_float=False)]
    u_ht_val=userRange('Enter your height (Range:0~300): ',0,300)
    u_wt_val=userRange('Enter your weight (Range:0~300): ',0,300)
    u_bmi_val=u_wt_val/((u_ht_val/100)**2) #BMI, cm->m
    u_wt=[u_wt_val]
    u_ht=[u_ht_val]
    u_bmi=[u_bmi_val] #list로 감싸주기
    u_wc=[userRange('Enter your waist circumference (Range:0~300): ',0,300)]
    while True:
        userKnow=input("AI has already replace question, So if you don't know your body information, You can replace your body info. Do you want to replace your info?(y|n)")
        if userKnow == 'n':
            print('You can enter your body info!')
            u_sbp=[userRange('Enter your systolic blood pressure(Range:30~300): ',30,300)]
            u_dbp=[userRange('Enter your diatolic blood pressure(Range:30~200): ',30,200)]
            u_glu=[userRange('Enter your glucose(Range:10~400): ',10,400)]
            break
        elif userKnow == 'y':
            u_sbp,u_dbp,u_glu = replace_user()
            print('Replaced!')
            break
        else:
            print('only [y|n]!')
    UserInfo = {
        'Gender':u_gender,
        'Age':u_age,
        'HT':u_ht,
        'WT':u_wt,
        'BMI':u_bmi,
        'WC':u_wc,
        'SBP':u_sbp,
        'DBP':u_dbp,
        'GLU':u_glu
    }
    print('-'*50)
    #result
    #vote model
    user_data=pd.DataFrame(UserInfo) #DF형태로 만들기
    #데이터 맞추기(X_all 기준),딕셔너리 데이터:9개,학습 데이터:9개
    match_data=user_data[['Gender','HT','WT','BMI','WC']]
    pred_user=optimal_vote_grid.predict(match_data)[0]
    proba_user=np.max(optimal_vote_grid.predict_proba(match_data)[0]) #사용자가 집단에 어느정도 속해있는지 확률
    #np.max:가장 높은 확률만 가져오기
    #reg
    pred_age=optimal_reg_grid.predict(user_data)[0]
    print('-User Status-') #군집화 정보로 학습->그러므로 그룹으로 상태 나타내기
    print(f'AI predicted user age: {pred_age:.0f}old')
    print(f'User belongs to {pred_user} Group: {proba_user*100:.2f}%')
    showgraph()
    print('-'*50)
    #시각화 자료
    print('Visualize: User and other patients information scatter and graph.')
    fig,axes=plt.subplots(1,2,figsize=(14,6)) #subplot으로 화면 분할,figsize로 사이즈 조절
    sns.countplot(x='positive_result',data=csv_edit,ax=axes[0]) #화면 분할을 위한 axes인덱스 번호 부여
    axes[0].set_xlabel('Positive|Negative Patients')
    axes[0].set_title('Paitents Information Graph')
    axes[0].set_xticks([0,1],['Negative','Positive'])
    sns.scatterplot(x='Age',y='GLU',data=csv_edit,color='red',marker='o',label='Patients',ax=axes[1],alpha=0.5) #나이와 대사증후군의 핵심인 GLU를 가져옴,[1]설정,투명도 0.5
    #사용자 정보
    scatter_age=user_data['Age'][0]
    scatter_glu=user_data['GLU'][0] #DF를 씌운 user_data에서 가져오기
    axes[1].scatter(x=scatter_age,y=scatter_glu,color='green',marker='*',label='User',zorder=5) #사용자 정보는 상단에 띄우기(zorder)
    axes[1].set_title('User and Patients Status (Metabolic Syndrome)')
    axes[1].legend()
    axes[1].axhline(126.0,color='blue',linestyle='--')#수평선
    plt.tight_layout()
    plt.show()
    while True:
        user_end=input('Restart this diagonsis? [y|n]')
        if user_end == 'y':
            break
        elif user_end == 'n':
            print('Test ended')
            break
        else:
            print('You can enter only [y|n]')
    if user_end == 'y':
        print('Restart!')
    elif user_end == 'n':
        break
# %%

