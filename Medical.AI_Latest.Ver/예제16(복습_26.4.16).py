#예제 16(HbA당화혈색소 측정) MAE(mean absolute error 평균 절대 오차),r2 score, Regressor
import pandas as pd
import os #sav->csv(metadata)
import pyreadstat as pt ##%%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_absolute_error, r2_score #(MAE,성적표)
from sklearn.impute import SimpleImputer

#sav,csv
sav_load = pt.read_sav('sav_file')
csv_file = pd.read_csv('csv_file')
if not os.path.exists (csv_file):
    sav_load,_=pt.read_sav('sav_file') 
    print('Converting...')
    sav_load.to_csv(csv_file,index=False,encoding='utf~') #check 
    #index = False: 내용이 없는 열에 번호 부여를 막을 수 있다. encoding=utf 한글 지원
    print('Convert')

#sav->csv (사용자가 이미 이름을 바꿨다고 가정(csv))
csv_load=pd.read_csv('csv_file')
#metadataonly
try:
    _,meta=pt.read_sav('sav_file',metadataonly=True) #틀은 제외하고 핵심내용인 Metadata만 챙긴다
    for c_name,c_label in meta.column_names_to_labels.items(): #제목,내용을 meta를 통해 가져온다
        label=c_label if c_label else 'Not explanation' #(설명=설명) <- 없는 알맹이들 걸러내기
        print('Access!')
    print('!')
except Exception as ex:
    print(f'ERROR{ex}')

print('Complete load file')

#train_test (Age,bmi,wc,glu,sbp,dbp|HbA)
X_all=csv_load[['Age','BMI','WC','GLU','SBP','DBP']]
Y_all=csv_load['HbA']
X_train,X_test,Y_train,Y_test=train_test_split(X_all,Y_all,test_size=0.25,random_state=42)

#doctor
doctor_reg=make_pipeline(
    SimpleImputer(strategy='median'), #결측치 중앙(평균을 낼 때 너무 차이가 심한 값 때문에 중앙으로 설정) 
    StandardScaler(),
    RandomForestRegressor(n_estimators=100,random_state=42) #나무 100개 설정, 난수 섞는 방법은 항상 42로 고정 
)
doctor_reg.fit(X_train,Y_train)

#MAE,R2
# MAE_X=mean_absolute_error(X_test)
# R2_X=r2_score(MAE_X)
#->
pred_X=doctor_reg.predict(X_test) #모델이 직접 풀어보는 과정(위에는 그게 없었음)
MAE_X=mean_absolute_error(Y_test,pred_X) #애초에 평균 오차를 내려면 비교할 대상이 있어야 하는데 아예 안 씀; -> Y_test와 직접 푼 문제를 비교
R2_X=r2_score(Y_test,pred_X) #직접 푼 문제랑 정답지 정확도를 나타내기

print('-'*50)
print(f'Mean Absolute Error: {MAE_X:.1f}')
if MAE_X <= 0.5:
    print('Good Model')
else: 
    print('Damn')

print(f'R2: {R2_X:.1f}') #전처리 뼈대 구축: 34분 내용 이해 했는지 주석을 보고 냉정한 평가.

#대체설문(구조만을 알기 위해 값과 변수명은 임의로 설정)
while True:
    def EstimateInfo():
        print('시작.') #최종 while 복귀 지점 지정
        #1.SBP,DBP
        u_sbp=0 #check
        u_dbp=0
        u_glu=0
        while True:
            SDbp = ('1. y|n')
            if SDbp == 'y':
                u_sbp += 100
                u_dbp += 100
                u_glu += 100
                break #왜 여기엔 빨간 밑줄이 생성되고 
            elif SDbp == 'n':
                u_sbp -= 100
                u_dbp -= 100
                u_glu -= 100
                break
            else:
                print('y|n !!')
        #2.GLU 
        while True:
            Glu = ('2. y|n')
            if Glu == 'y':
                #u_sbp += 100 #check BP구역이 아니기에 관련 코드를 쓰면 안됨
                #u_dbp += 100
                u_glu += 100
                break #왜 여긴 생성이 안될까? 둘이 구조는 똑같음
            elif Glu == 'n':
                #u_sbp -= 100
                #u_dbp -= 100 
                u_glu -= 100
                break
            else:
                print('y|n !!')
        #3.미세조정
        while True:
            U_info = ('3. y|n')
            if U_info == 'y':
                u_sbp += 100
                u_dbp += 100
                #u_glu += 100 <- check
                break #왜 여긴 생성이 안될까? 둘이 구조는 똑같음
            elif U_info == 'n':
                u_sbp -= 100
                u_dbp -= 100
                #u_glu -= 100
                break
            else:
                print('y|n')
        #check return, 추가된 정보를 가진 변수명들을 return
        return u_sbp, u_dbp, u_glu
    ReplaceInfo = EstimateInfo()

    #대체설문 vs 직접 쓰기
    def UserInfo(prompt,minVal,maxVal,Fl0at=True):
        try:
            Uvalue = float(input(prompt)) if Fl0at else int(input(prompt))
            if minVal <= Uvalue <= maxVal:
                return(Uvalue)
            print('what?')
        except ValueError:
            print('ERROR')
    while True:
        UserYesNo=input('대체y|직접n')
        if UserYesNo == 'y':
            print('대체')
            u_sbp, u_dbp, u_glu = ReplaceInfo #check 추정정보 함수에서 썼던 변수들을 ReplaceInfo에 넣어야함
            break
        elif UserYesNo == 'n':
            print('직접')
            u_sbp = UserInfo('sbp: 0~500',0,500)
            u_dbp = UserInfo('dbp: 0~500',0,500)
            u_glu = UserInfo('glu: 0~500',0,500)
            break
        else:
            print('y|n!!')
    
    #여긴 직접 입력
    u_age = UserInfo('age: 0~100',0,100,Fl0at=False)
    u_wc = UserInfo('wc: 0~200',0,200)
    u_bmi= UserInfo('bmi: 0~50',0,50)

    #check 사전에다가 pd.DataFrame적용x
    UserType={ #왜 불이 안 들어오지? -> 들여쓰기의 늪에 빠짐..(위에 잘 보면서 하기)
        'Age':[u_age], 
        'BMI':[u_bmi],
        'WC':[u_wc],
        'GLU':[u_glu],
        'SBP':[u_sbp],
        'DBP':[u_dbp]
    }

    #사용자 기반 
    UserFrame=pd.DataFrame(UserType) #불이 안 들어오니 여기도 못 받음
    pred_u=doctor_reg.predict(UserFrame)[0]
    if pred_u < 6.0:
        print(f'HbA: {pred_u:.1f}%')
    elif 6.0 <= pred_u <= 6.4:
        print(f'HbA: {pred_u:.1f}%')
    elif 6.5 <= pred_u:
        print(f'HbA: {pred_u:.1f}%')
    else:
        print(f'HbA: {pred_u:.1f}%')

    while True:
        YesNo = input('재시작? y|n')
        if YesNo == 'y':
            print('재개')
            break
        elif YesNo == 'n':
            print('종료')
            break
    if YesNo == 'n': #check 
        break #최종 1시간 28분 코드 작성 끝