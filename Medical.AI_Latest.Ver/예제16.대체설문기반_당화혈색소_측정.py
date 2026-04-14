#본 EMR데이터는 XX대학교  극히 일부분 '틀'(내용X)을 가지고 진행했음을 알립니다
# 절대 실제 환자의 기록을 토대로 학습을 하지 않았으며 이를 가지고 유포하거나 타인에게 공유할 시, 국내 의료법에 위반된다는 걸 인지하고 있습니다.
#%%
import pandas as pd
import os
import pyreadstat as pt
from sklearn.impute import SimpleImputer #결측값(공백)을 채워주는 역할 계산에 오류가 안 생기게 방지를 해줌
#MAE(Mean_Absolute_Error) 평균 절대 오차
#만약 오차 범위가 0.2가 나왔을 경우: 벗어난 범위가 매우 적으며 충분히 쓸만한 모델이라고 평가됨
#r2_score: 결정계수, 모델의 '성적표', 1.0은 만점이고, 0.9면 쓸만한 모델이라고 보면 된다. 
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#회귀는 '예측'모델이기에 MAE를, 분류는 '분류'모델이기에 accuracy를
#예측을 하면 값이 5.n~ 씩으로 나오기에 (0,1,2같이 나오는게 아님) MAE를 써야 함
from sklearn.ensemble import RandomForestRegressor 
from sklearn.pipeline import make_pipeline 

# #load sav (sav->csv) 만약 csv파일이 없을 경우 (코드 안 까먹었는지 확인)
# sav_load=pt.read_sav('SpssTraining[sav]/training.sav')
# if not os.path.exists (sav_load):
#     print('Converting sav->csv...')
#     sav_load.to_csv(sav_load, index = 0, encoding = '한글 지원 코드')
#     print('Complete Converting!')

#load csv
csv_load = pd.read_csv('SpssTraining[csv]/training.csv')
# print('Model is extracting only metadata...')
# print('-'*50)

# #csv(metadataonly)
# try: 
#     _,meta=pt.read_sav(sav_load,metadataonly=True) #metadataonly(잡다한 거 싹 빼고 설명서만!)
#     for c_name,c_label in meta.columns_to_labels.items(): #for (name,explaination)
#         label = c_label if c_label else 'Not exist description'
#         print('Complete extracting csv file!')
# except Exception as ex:
#     print(f'ERROR: {ex}')

print('Complete load!')

#train_test (age,bmi,wc,glu,sbp,dbp|HbA) WC:허리둘레, HbA:당화혈색소
X_reg=csv_load[['Age','SBP','DBP','WC','BMI','GLU']] #공부 자료
Y_reg=csv_load['HbA'] #답지
X_train_reg,X_test_reg,Y_train_reg,Y_test_reg=train_test_split(X_reg,Y_reg,test_size=0.25,random_state=42)

#pipeline
HbA_doctor=make_pipeline(
    SimpleImputer(strategy='median'),
    #결측값은 중앙(median)으로! -> 높은 값의 존재로 평균이 이상하게 바뀔 수 있기에, 가장 중앙의 값을 기준으로 설정
    StandardScaler(),
    RandomForestRegressor(n_estimators=400,random_state=42)
    #무작위 나무 400개 생성(자료가 많을 시 사용자가 조정)
    #random_state로 섞는 방법 고정(이래야 언제든 모델을 돌려도 계산이 똑같음)
)
HbA_doctor.fit(X_train_reg,Y_train_reg)
#predict
pred_X=HbA_doctor.predict(X_test_reg)

#MAE(Mean_Absolute_Error) 평균 절대 오차
MAE_X=mean_absolute_error(Y_test_reg,pred_X) #평균에서의 오차 테스트
#말 그대로 평균에서의 오차이기에 테스트를 해야한다 
R2=r2_score(Y_test_reg,pred_X) #모델의 정확성(설명력)
#(실제 답안,모델이 푼 답안)

print('-'*50)
print(f'MAE(평균 절대 오차): {MAE_X:.2f} %')
print(f'R2(모델의 설명력){R2:.2f}')
print('해석: MAE가 0.5라면 모델이 예측한 수치가 실제 환자의 HbA 평균 +-0.5% 정도만 차이 난다는 겁니다. ')
print('-'*50)

#대체설문기반 (bp,glu,미세조정)
while True:
    def EstimateInfo():
        print('건강 설문(y|n으로만 답해주세요!)')
        #1.BP
        while True:
            s_d_bpQ=input('Q1.과거에 고혈압 진단을 받았거나, 현재 복용하는 혈압약이 있나요?(y|n)')
            if s_d_bpQ == 'y':
                #수축,이완기 나눠주기(고혈압 수치 부여)
                est_sbp = 145.0
                est_dbp =95.0
                print('질문이 입력됐습니다!')
                break
            elif s_d_bpQ == 'n':
                #수축,이완기 나눠주기(정상 수치 부여)
                est_sbp = 120.0
                est_dbp =80.0
                print('질문이 입력됐습니다!')
                break
            else:
                print('y,n으로만 대답해주세요')
        #2.GLU
        while True:
            gluQ=input('최근 건강검진에서 "혈당 높음"이라는 진단을 받은 적이 있나요?(y|n)')
            if gluQ == 'y':
                est_glu=126.0
                print('질문이 입력됐습니다!')
                break
            elif gluQ == 'n':
                est_glu = 95.0
                print('질문이 입력됐습니다!')
                break
            else:
                print('y,n으로만 대답해주세요')
        #3.미세조정
        while True:
            user_q=input('일주일에 3회 이상 땀이 나는 운동을 하나요? (y|n)')
            if user_q == 'y':
                est_sbp -= 5.0
                est_dbp -= 5.0
                print('질문이 입력됐습니다!')
                break
            elif user_q == 'n':
                est_sbp += 5.0
                est_dbp += 5.0
                print('질문이 입력됐습니다!')
                break
            else:
                print('y,n으로만 대답해주세요')
        return est_sbp,est_dbp,est_glu
    #알면 직접 입력, 모르면 질문 받아서 데이터 입력
    def u_range(prompt,minVal,maxVal,fl0at=True):
        while True:
            try:
                    Uvalue= float(input(prompt)) if fl0at else int(input(prompt))
                    if minVal <= Uvalue <= maxVal:
                        return(Uvalue)
                    print('over range!')
            except Exception as ex:
                print(f'ERROR:{ex}')
    while True:
        #돌아갈 곳을 만들자
        userQ=input('사용자는 본인의 정보를 직접 알고 있으면 y, 모르면 n을 눌러주세요: ')
        if userQ == 'y':
            print('입력됐습니다!')
            u_sbp = u_range('수축기 혈압(SBP) 입력(Range:30~300):',30,300)
            u_dbp = u_range('이완기 혈압(DBP) 입력(Range:30~200):',30,200)
            u_glu = u_range('공복혈당(GLU) 입력(Range:10~400):',10,400)
            break
        elif userQ== 'n':
            print('입력됐습니다!')
            #정보 받기
            u_sbp, u_dbp, u_glu = EstimateInfo()
            break
        else:
            print('y|n으로만 대답해주세요')

    u_age=u_range('나이를 입력해주세요(Range:0~100)',0,100,fl0at=False)
    u_wc=u_range('허리 둘레 입력해주세요(Range:30~150)',0,150)
    u_bmi=u_range('bmi 입력해주세요(Range:10~50)',10,50)

    UserInfo={ #dictionary
        'Age':[u_age],
        'SBP':[u_sbp],
        'DBP':[u_dbp],
        'WC':[u_wc],
        'BMI':[u_bmi],
        'GLU':[u_glu]
    } 
    user_data=pd.DataFrame(UserInfo)
    user_HbA=HbA_doctor.predict(user_data)[0] #<-사용자 HbA값 나타내기
    if user_HbA < 6.0:
        print(f'사용자 당화혈색소 범위: {user_HbA:.1f}')
        print('정상입니다')
    elif 6.0 <= user_HbA <= 6.4:
        print(f'사용자 {user_HbA:.1f}')
        print('당뇨 의심 단계입니다')
    elif user_HbA >= 6.5:
        print(f'사용자 당화혈색소 범위: {user_HbA:.1f}')
        print('당뇨입니다')
    while True:
        YesNo = input('테스트를 다시 진행할까요? (y|n)')
        if YesNo == 'y':
            print('재시작!')
            break #이 방 터트리고 위로 가기
        elif YesNo == 'n':
            print('종료')
            break #터트리고 아래로 가기
        else:
            print('y|n만 입력')
    if YesNo == 'n':
        break #마지막으로 다 터트리고 끝