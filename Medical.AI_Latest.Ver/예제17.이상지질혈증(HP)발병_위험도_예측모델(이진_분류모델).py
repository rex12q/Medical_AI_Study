#본 EMR데이터는 XX대학교  극히 일부분 '틀'(내용X)을 가지고 진행했음을 알립니다
# 절대 실제 환자의 기록을 토대로 학습을 하지 않았으며 이를 가지고 유포하거나 타인에게 공유할 시, 국내 의료법에 위반된다는 걸 인지하고 있습니다.
#%%
import seaborn as sns
#seaborn: matplotlib 기반 업그레이드 통계 특화 버전(시각적 자료 퀄UP)
import matplotlib.pyplot as plt
import pandas as pd
import os
import pyreadstat as pt
from sklearn.preprocessing import LabelEncoder
#LabelEncoder: 문자의 형식을 숫자로 변환해주는 역할, 모델은 '숫자'만 읽을 수 있기에 변환을 해줘야 한다.(전처리 과정)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier #예측 판독기
from sklearn.impute import SimpleImputer #결측치 처리
from sklearn.metrics import accuracy_score, f1_score
#f1_score(정밀도):정확성 오류(뻥튀기 현상)를 잡아내기 위해서 f1을 사용함
#예)99명 맞췄고 1명 틀렸다고 해서 99프로가 나오는 이런 뻥튀기 현상을 잡기 위한 안전 장치
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#sav,csv (sav->csv를 한다는 가정하에)|(암기 목적으로 직접 다 타이핑)
sav_file = 'training.sav'
csv_file = 'training.csv' #확장자를 바꾸기 -> to_csv를 통해 같이 바뀜

if not os.path.exists(csv_file):
    d_sav,_= pt.read_sav(sav_file)
    print('Converting...')
    d_sav.to_csv(csv_file,index=False,encoding='utf-8-sig')
    print('Complete converting!')

#metadata
try:
    _,meta = pt.read_sav(sav_file,metadataonly=True)
    for c_name,c_label in meta.column_names_to_labels.items():
        label=c_label if c_label else 'Not Explaination'
        print(f'Explaination: {label}')
        print('Complete load!')
except Exception as e:
    print(f'ERROR: {e}')

#csv
csv_load = pd.read_csv('csv/[csv]training(HP전용).csv') #csv 파일 안에

#굳이 LabelEncoder와 get_dummies를 쓰기(모르는 개념이니 공부)
# #해당 sav파일은 인코딩이 되어있는 상태지만 학습을 위해 바꿔보자
# def make_bmi(bmi):
#     if bmi < 18.5:
#         return '저체중'
#     elif bmi < 25.0:
#         return '정상'
#     else:
#         return '비만'

# #BMI열에 BMI_글자라는 열을 함수(make_bmi)값을 적용해 원본 sav파일의 새로운 열로 자리를 잡음
# csv_load['BMI_글자'] = csv_load['BMI'].apply(make_bmi)
# #번호 부여과정
# le=LabelEncoder()
# #'BMI_라벨'이라는 새로운 열로 인코딩한 결과들을 담아주기
# csv_load['BMI_라벨']=le.fit_transform(csv_load['BMI_글자'])
# #이렇게 되면 저체중:0,정상:1,비만:2가 '라벨'열에 들어가게 됨

# #pd.get_dummies 원~핫 코딩(열 안에 있는 내용물을 '열'형태로 찢게 하기)
# csv_load=pd.get_dummies(csv_load, columns=['BMI_글자'])
# #이렇게 되면 원본 파일을 담고 있는 csv_load에 'BMI_글자'가 글자_저체중,_정상,_비만 형태로 찢어지게 된다
# # 여기서 사용자 정보 기반(예:사용자는 저체중)으로 모델을 돌리면 _저체중:1,_정상:0,_비만:0으로 열 형태가 유지가 된다.  

#encoding:모델은 무조건 숫자만 읽는다 그러니 형식을 (문자->숫자)로 바꿔줘야 한다
# (0|1)<-이진 분류(a/stype(int))로 나타내기
#이상지질혈증GS(HP),이상지질혈증KIT(DSP)|
# pos_hp = (csv_load['?'] >= 0.0)
# csv_load['hp_target'] = pos_hp.astype(int) # True: 1, False: 0 규칙 만들기(추가로 새로운 열도 만들기)

#이상지질혈증GS(HP)
#원본 파일 덮어주기(원본: 870 -> 수정:842, 결측치:28)
csv_load=csv_load.dropna(subset=['HP'])
csv_load['hp_target']=csv_load['HP']
pos_hp=csv_load['hp_target']
#사실 굳이 열 하나를 더 추가 안 해도 되지만 학습을 위해..
#dropna: drop+nan(결측치) 결측치가 존재하는 행을 삭제
#subset: '부분 집합'을 의미, 사용자가 선택한 행만 삭제|columns,subset차이->열과 행
#dropna columns는 그냥 카테고리 통으로 없애는 거니깐 쓰면 안됨| (drop,columns),(dropna,subset)

#train_test
#1.기본 인적 사항 (성별,나이,나이그룹): 호르몬 영향 콜레스트롤 수치 변화 양상
#1-2.기본 인적 사항 (사는 지역, 교육 수준): 환경적 요인, 거주 지역에 따른 운동과 같은 활동량 확인
X_info=csv_load[['Gender','Age','AgeGroup','Town','Education']] 
#2.(body=by)신체 계측 지표(겉보기 변화),(신장,허리둘레,체질량지수):허리둘레가 두껍고 체질량 지수가 높으면 몸에 지방이 낀 경우가 대다수
X_by=csv_load[['HT','WC','BMI']]
#3-1.(blood=bd)대사 증후군 지표(다른 장기나 혈관은 무사한가),(수축기,이완기):혈관에 기름이 끼면 피를 보내야 하는데 이 때가 강도 높은 펌프질로 인해 혈압 상승
#3-2.대사 증후군 지표(공복혈당,당화혈색소,당뇨병여부):당을 잘 분해하지 못한 경우(당뇨), 당이 몸 안에 남게 되어 지방으로 변함, 혈당 수치가 높으면 이상지질혈증 위험도도 높아짐
X_bd=csv_load[['SBP','DBP','GLU','HbA','DM']]
#4.답(이상지질혈증)
Y_hp=csv_load['hp_target']
#묶기(concat으로 붙이기 -> 하나의 표로 만들기)
# X_all=pd.concat([X_info,X_by,X_bd],axis=1)
#concat: 기존에 있었던 표를 (옆,아래로)붙어서 하나의 거대한 표로 만드는 과정 
#axis=1:행의 형태를 유지하되, 특성(열)들을 맞춰서 옆으로 쭉 이어 붙이는 것. (0일 경우, 세로)

(X_train_info,X_test_info,
X_train_by, X_test_by,
X_train_bd, X_test_bd,
Y_train,Y_test
) = train_test_split(
    X_info,X_by,X_bd,Y_hp,test_size=0.2,random_state=42) #한 바구니에 2개씩

# X_train_all,X_test_all,Y_train,Y_test=train_test_split(X_all,Y_hp,test_size=0.2,random_state=42)

#X에는 정보가 없음
# X_train_hp = X_train_all.drop(columns=['hp_target'])
# X_test_hp = X_test_all.drop(columns=['hp_target']) #참고로 정답이 있는 Y는 drop(x)

#모델 및 학습
doctor_info = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    RandomForestClassifier()
)
doctor_by = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    RandomForestClassifier()
)
doctor_bd = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    RandomForestClassifier()
)

#SimpleImputer는 X의 결측치만 대체하지, Y는 못 건든다->상식적으로 생각하면 잘못된 값을 모델한테 준다는 소리인데 이러면 말이 안된다
doctor_info.fit(X_train_info,Y_train)
doctor_by.fit(X_train_by,Y_train)
doctor_bd.fit(X_train_bd,Y_train)

#정확도 및 Soft Voting 설계(정보,몸,혈액) 양성일 확률만 추리기
#[:,1]은 1(양성)일 확률만 뽑는 것 <-> [:,0]은 0(음성)일 확률만 뽑는 것
#1.정보
pred_info=doctor_info.predict(X_test_info)
acc_model=accuracy_score(pred_info,Y_test)
f1=f1_score(pred_info,Y_test)
prob_info=doctor_info.predict_proba(X_test_info)[:,1] 

#2.몸
pred_by=doctor_by.predict(X_test_by)
acc_model=accuracy_score(pred_by,Y_test)
f1=f1_score(pred_by,Y_test)
prob_by=doctor_by.predict_proba(X_test_by)[:,1]

#3.혈액
pred_bd=doctor_bd.predict(X_test_bd)
acc_model=accuracy_score(pred_bd,Y_test)
f1=f1_score(pred_bd,Y_test)
prob_bd=doctor_bd.predict_proba(X_test_bd)[:,1]

#합치기(Soft Voting) 
prob_all=(prob_info+prob_by+prob_bd)/3
prob_final=(prob_all>=0.5).astype(int) #최종 확률이 0.5이상이면 1, 아니면 0 
acc_model=accuracy_score(prob_final,Y_test)
f1=f1_score(prob_final,Y_test)

print('-'*50)
print(f'Model [Users information part] accuracy: {acc_model*100:.2f}%')
print(f'Model [Users information part] precision: {f1*100:.2f}%') 
print('-'*50)
print(f'Model [Users body part] accuracy: {acc_model*100:.2f}%')
print(f'Model [Users body part] precision: {f1*100:.2f}%')
print('-'*50)
print(f'Model [Users blood part] accuracy: {acc_model*100:.2f}%')
print(f'Model [Users blood part] precision: {f1*100:.2f}%') 
print('-'*50)
print(f'Model total accuracy: {acc_model*100:.2f}%')
print(f'Model total precision: {f1*100:.2f}%') #정밀도(ML이 진짜라고 고른 것 중 진짜만 추렸을 때)
# if prob_final == 1: #양성일 확률만 추렸을 때 모델 성능이 좋으면 1이 나올 것이고 안 좋으면 0
#     print(f'This sav file contained percentage of positive patient: {prob_all*100:.2f}%')
# elif prob_final == 0:
#     print(f'This sav file contained percentage of positive patient: {1-prob_all*100:.2f}%')
# else:
#     print("Model can't measure stats") #한 명을 대상으로 모델을 돌리는게 아니기에 주석처리
print('Complete load from model')
#1.Info
def Info(prompt,q_type,choices=None,minVal=0,maxVal=150,is_int=True):
#choices(None):선택 종류,(choices를 안 쓸 때는 그냥 없는 매개변수), q_type:질문 유형, min,max: 매개변수와 변수의 수가 같아야 하지만 값을 부여하므로 이를 방지
    while True:
        try:
            userVal = input(prompt).strip()#dictionary기준에 맞춰진다
            #공장1. choice의 경우
            if q_type == 'choice': #질문 유형 'choice'인 경우
                if userVal in choices: # 사전을 기준으로 해서 정보 입력
                    return choices[userVal]
                print(f'True answer: {list(choices.keys())}') #사전 리스트 정보 불러오기
            #공장2. range의 경우
            elif q_type == 'range':
                #if userVal in choices: 애초에 choices는 없음
                userNum = int(userVal) if is_int else float(userVal)
                #prompt를 바꾸는게 아닌 userVal을 바꿔야 함
                if minVal <= userNum <= maxVal:
                    return userNum
                print(f'Over range! | Range: {minVal}~{maxVal}')
        except ValueError:
            print('Oops! try again')
    
#choice:객관식, range:주관식
#u_gen 같은 경우 Info 함수로 묶었으면 매개변수의 수가 맞아야 하지만 맞지 않기에 에러가 뜬다->(조치: 매개변수에 기본값을 부여)
u_gen = Info(
    prompt = 'What is your gender? (male|female):',
    q_type = 'choice',
    choices = {'male':1, 'female':2}, #dictionary 
)
u_age = Info(
    prompt = 'How old are you? (Range: 0~150) :',
    q_type = 'range',
    minVal=0,maxVal=150 #조건에 쓰려는 값이 함수 매개변수의 값과 다를 경우 사용자가 바꿀 수 있다.
)
u_town = Info(
    prompt = 'Where are you live now? (1:동|2:읍면)',
    q_type = 'choice',
    choices = {'동':1, '읍면':2}, #dictonary
)
u_edu = Info(
    prompt = 'What is your the highest of university',
    q_type = 'choice',
    choices = {'E':1,'M':2,'H':3,'U':4}, #dictonary
)
#2.Body
#bmi는 wt,ht로 의학 공식 계산으로 값을 출력
u_wt = Info(
    prompt = 'Enter your weight(WT)',
    q_type = 'range',
    minVal = 10, maxVal = 250,
    is_int = False
)
u_ht = Info(
    prompt = 'Enter your height(HT):',
    q_type = 'range',
    minVal = 100, maxVal = 250,
    is_int = False
)
u_wc = Info( #남성 기준
    prompt = 'Enter your waist circumference(WC):',
    q_type = 'range',
    minVal = 40,
    is_int = False
)
u_bmi = u_wt / ((u_ht/100)**2) # 계산 공식: 몸무게/키(100을 곱한 상태에서 제곱)
# u_bmi = Info( 사용자가 임의로 bmi를 넣기 보단 계산 공식을 통해 bmi값을 출력
#     prompt = 'Enter your bmi:',
#     q_type = 'range',
#     minVal = 10 ,maxVal = 50,
#     is_int = True
# )
#3.Blood
u_sbp = Info(
    prompt = 'Enter your systolic blood pressure:',
    q_type = 'range',
    minVal = 40 ,maxVal = 250,
    is_int = False
)
u_dbp = Info(
    prompt = 'Enter your diatolic blood pressure:',
    q_type = 'range',
    minVal = 40 ,maxVal = 180,
    is_int = False
)
u_glu = Info(
    prompt = 'Enter your glucose:',
    q_type = 'range',
    minVal = 50 ,maxVal = 170,
    is_int = False
)
u_hba = Info(
    prompt = 'Enter your HbA:',
    q_type = 'range',
    minVal = 2 ,maxVal = 15,
    is_int = False
)
u_dm = Info(
    prompt = 'Enter your diabetes target(0:negative|1:positive)',
    q_type = 'choice',
    choices = {'0':0,'1':1},
    is_int = True
)
#조건
#gen,wc(의학적 자료에 의하면 남자,여자 허리둘레 정상,비만 기준이 다름)
if u_gen == 1: # Male
    if u_wc >= 90:
        print("[Warning] Abdominal obesity detected. (High risk of metabolic syndrome)")
    else: #복부 지방| 대사 증후군: 신진대사(대사)와 관련된 질환이 동반된다는 의미
        print("Normal waist circumference.")
elif u_gen == 2: # Female
    if u_wc >= 85:
        print("[Warning] Abdominal obesity detected. (High risk of metabolic syndrome)")
    else: #복부 지방이 발견됨. (대사 증후군 고위험)
        print("Normal waist circumference.") 
        #평범한 허리 둘레
#bmi 
if u_bmi >= 35.0:
    print(f"[Danger] BMI: {u_bmi:.1f} - Obesity Class 3 (Extremely high risk). Immediate consultation recommended.")
elif u_bmi >= 30.0: #비만 3단계 (매우 위험). 즉시 상담을 추천
    print(f"[Warning] BMI: {u_bmi:.1f} - Obesity Class 2 (High risk).")
elif u_bmi >= 25.0: #비만 2단계 (고위험)
    print(f"[Caution] BMI: {u_bmi:.1f} - Obesity Class 1 (Moderate risk).")
elif u_bmi >= 23.0: #비만 1단계 (중위험)
    print(f"[Notice] BMI: {u_bmi:.1f} - Overweight (Pre-obesity stage).")
elif u_bmi >= 18.5: #과체중 (비만 전 단계)
    print(f"BMI: {u_bmi:.1f} - Normal weight.")
else: #정상 체중
    print(f"[Caution] BMI: {u_bmi:.1f} - Underweight. Nutritional management may be needed.")
    #저체중. 영양 관리가 사용자에게 도움이 될 듯

#blood pressure 
if u_sbp >= 160 or u_dbp >= 100:
    print("[Danger] Stage 2 Hypertension (High risk).")
elif u_sbp >= 140 or u_dbp >= 90: #고혈압2 매우 위험
    print("[Warning] Stage 1 Hypertension. Blood pressure management is required.")
elif u_sbp >= 130 or u_dbp >= 80: #고혈압1 혈압 관리가 요구됨
    print("[Caution] Elevated Blood Pressure (Prehypertension).")
elif u_sbp < 120 and u_dbp < 80: #전고혈압 주의 혈압
    print("Optimal normal blood pressure.")
else: #최적 혈압
    print("[Notice] Blood pressure is within the caution range.")
        #혈압이 주의 범위 안에 있음

#glucose 
if u_glu >= 126:
    print("[Danger] Suspected Diabetes. glucose is critically high.")
elif u_glu >= 100: #당뇨 의심, 글루코스가 매우 치명적
    print("[Caution] Impaired Glucose (Pre-diabetes). Control your sugar intake.")
else: #손상된 글루코스(전 당뇨),사용자는 당분 섭취를 조절
    print("Normal blood glucose.")
    #정상 공복 혈당
#HbA(당화혈색소)
if u_hba >= 6.5:
    print("[Danger] Diabetes diagnosis range. Medical treatment is highly advised.")
elif u_hba >= 5.7:#당뇨 진단 범위, 의학적 치료를 강하게 조언
    print("[Caution] Pre-diabetes range. Change your lifestyle.")
else:#전 당뇨 단계, 라이프 스타일 바꾸기
    print("Normal HbA1c level.")
    #평범한 당화혈색소 단계
# DM(diabetes mellitus:당뇨병)
if u_dm == 1:
    print("[Record] Prior history of diabetes noted. This will strongly affect the prediction.")
else:#이전 당뇨 진료에 기록됨. 이것은 예측에 강하게 영향이 감
    print("[Record] No prior history of diabetes.")
    #당뇨에 관한 이전 진료 기록이 없음

#13개의 변수 맞추기, PandasDataFrame으로 맞춰주기 | 위와 똑같이 info,body,blood로 나눠주기
u_ageg = u_age//10 #AgeGroup /:flaot //:int
UserInfo = {
    'Gender':[u_gen],
    'Age':[u_age],
    'AgeGroup':[u_ageg],
    'Town':[u_town],
    'Education':[u_edu]
}
UserBody = {
    'HT':[u_ht],
    'WC':[u_wc], #ui 파트에서 변수를 굳이 맞출 필요는 없다. 사용자 정보를 받을 딕셔너리는 개수가 정확해야 함.
    'BMI':[u_bmi]
}
UserBlood = {
    'SBP':[u_sbp],
    'DBP':[u_dbp],
    'GLU':[u_glu],
    'HbA':[u_hba],
    'DM':[u_dm]
}
#DF fit했을 때 액셀 표 형태로 학습을 시켰기에 딕셔너리를 주지 말고 df로 변환해서 하기
df_UserInfo = pd.DataFrame(UserInfo)
df_UserBody = pd.DataFrame(UserBody)
df_UserBlood = pd.DataFrame(UserBlood)

#피날레(각 파트 예측 확률 출력 및 합쳐서 최종 출력) f:finsh| [:,1]: 양성일 확률만 출력
UserInfof = doctor_info.predict_proba(df_UserInfo)[:,1]
UserBodyf = doctor_by.predict_proba(df_UserBody)[:,1]
UserBloodf = doctor_bd.predict_proba(df_UserBlood)[:,1] #df로 감싸준 자료 삽입

UserTotal = (UserInfof + UserBodyf + UserBloodf / 3)
UserProb = UserTotal[0] #[0]사용자 정보

print('-'*50)
print("EMR patients 'HP' information")
print('-'*50)
#seaborn
sns.countplot(x='hp_target', data=csv_load)
#countplot: seaborn내장어 기능 중 하나, 그룹별로 묶어서(ex:0,1)개수를 세서 막대그래프로 표현
plt.title("EMR patients 'HP' information")
plt.show()

print('-'*50)
print('Medical AI analysis complete!')
print('-'*50)
#확률 [:,1]로 계산되었기에 모든 확률은 양성 확률 가능성에 기준임
if UserProb >= 0.5:
    print(f"[Result] Dyslipidemia Risk Detected! (Predicted: 1)")
    #결과: 고지혈증(Dyslipidmia) 발견 (양성)
    print(f"Your calculated risk probability is {UserProb * 100:.1f}%.")
    #사용자의 양성 위험 가능성은 UserProb %로 계산됨
    print("Medical AI highly recommend consulting with a healthcare professional for a detailed check up and lifestyle management.")
    # 의료 ai는 정밀 검진과 라이프 스타일 관리를 위해 헬스케어 전문가와 함께 상담을 받아보는 것을 강력히 추천
else:
    print(f"[Result] Normal / Low Risk (Predicted: 0)")
    #결과: 정상, 낮은 위험 (음성)
    print(f"Your calculated risk probability is {UserProb * 100:.1f}%.")
    #사용자의 양성 위험 가능성은 UserProb %로 계산됨
    print("Keep up the good work. Maintain your current healthy diet and exercise routine.")
    #계속 좋은 상태 유지. 사용자는 건강 다이어트와 운동 루틴을 계속 유지 ㄱㄱ
# %%
#end
#의학 지식(기준은 위에)
# 1.WC(Waist Circumference)허리 둘레 - 대사 증후군(metabolic syndrome) 판정의 핵심
# 성별에 따른 기준이 다르며 기준을 넘어가면 '복부 비만 '으로 대사증후군 위험군에 들어감
# 2.SBP(Systolic Blood Pressure)/DBP(Diatolic Blood Pressure) - 대한고혈압학회 기준
# 혈압은 수축기와 이완기 중 더 높은 단계에 속하는 수치를 기준으로 판정함 
# (정상->주의혈압->고혈압 전단계->1기->2기)
# 3.GLU(Glucose)공복 혈당 - 대한당뇨병학회 기준
# 최소 8시간 이상 금식 후 측정한 혈당 기준
# (정상->공복혈당장애 1단계 - 당뇨 전단계->당뇨병 의심 고위험 2단계)
# 4.HbA(Hemoglobin)당화혈색소 - 당뇨 판정의 절대 기준
# 수명이 2~3개월인 적혈구에 설탕이 붙고 한 번 붙은 설탕은 적혈구가 죽을 때까지 계속 붙어있음, 최근 2~3개월간의 평군 혈당 상태를 보여주는 중요한 지표.
# 전체 적혈구에서 설탕이 묻어있는 비율을 나타냄
# (정상->당뇨 전단계->당뇨병 확진)
# DM(Diabetes Mellitus) 당뇨병
# 과거에 의사에게 당뇨병 진단을 받은 적이 있는 가를 묻는 지표
# (0:음성,1:양성)