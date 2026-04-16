#본 EMR데이터는 XX대학교  극히 일부분 '틀'(내용X)을 가지고 진행했음을 알립니다
# 절대 실제 환자의 기록을 토대로 학습을 하지 않았으며 이를 가지고 유포하거나 타인에게 공유할 시, 국내 의료법에 위반된다는 걸 인지하고 있습니다.
#%%
import pandas as pd
import pyreadstat as pt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#sav->csv 변환, 
SavCsv='SpssTraining[sav]/training.sav'
CsvSav='SpssTraining[csv]/training.csv' #이름,확장자 csv로 바꾸기

if not os.path.exists(CsvSav):
    sav_load,_=pt.read_sav('SpssTraining[sav]/training.sav')
    #찐데이터만 뽑아오기 위해 '_'로 메타데이터(설명서)를 drop하고 csv파일을 생성
    sav_load.to_csv(CsvSav, index =False, encoding='utf-8-sig')
    #CsvSav를 csv로 만들기 위해 이름을 바꾸고 인덱스(쓸데없는 것까지 번호 부여 방지), 엔코딩은 글자 깨짐 방짐을 위해 설정
    print('변환 성공')

#csv 불러오기
csv_load=pd.read_csv('SpssTraining[csv]/training.csv')

#csv파일 토대로 주피터 실행
try:
    Csv_df=pd.read_csv('SpssTraining[csv]/training.csv')
    #jupyter를 통해 셀 실행을 한 후 DF(찐 내용물),metadata(각 데이터 항목) 이렇게 나뉨| 여기서 meta를 추가해서 tuple에서 DF로 바뀜
    print(f'출력.\n {Csv_df}')
    print('-'*50)
    print('데이터 프레임 상세 설명')
    _,meta = pt.read_sav('SpssTraining[sav]/training.sav', MetaDataOnly=True)
    for ColName,ColLabel in meta.columns_names_to_labels.item():
        #meta~lables는 sav파일을 읽기 위한 거대한 사전(내장어)(순서: 이름,설명)|itmes():내용 가져오기
        label = ColLabel if ColLabel else '설명 없음'
        #사용자가 보고 싶어하는 label칸에 c_label(설명)을 넣어서 보여줌| 없는 경우: "설명 없음"으로 대체 
        print(f'[설명]->{label}')
except Exception as ex:
    print(f'에러 발생: {ex}')

print('파일을 불러왔습니다.')
#pyreadstat을 이용해서 sav불러온다

#%%
#ML Time! id,age,gender,sbp,dbp,bmi,glu 나이 예측, 당뇨 판정

#전체
#X_cla1=Csv_df["Age","SBP","DBP","BMI","GLU"]

#diabets_target
d_risk=[]
for d in range (len(Csv_df)):
    if Csv_df["SBP"][d]>=120 and Csv_df["DBP"][d]>=90  and Csv_df["BMI"][d] >= 25.5 and Csv_df["GLU"][d] >= 126:
        d_risk.append(1)
    else:
        d_risk.append(0)

Csv_df["diabetes_target"] = d_risk #csv_df에 드감

X_cla=Csv_df[["Age","SBP","DBP","BMI","GLU"]] #2nd array
#1. X_cla2=X_cla1["Age","SBP","DBP","BMI","GLU","diabetes_target"]
#2. X_cla2=X_cla1["diabetes_target"]

#학습,시험 부분
Y_cla = Csv_df["diabetes_target"] #당뇨 테스트
Y_reg = Csv_df["Age"] #나이 예측를 위한
#학습,시험량 나누기
X_train_cla,X_test_cla,Y_train_cla,Y_test_cla,Y_train_reg,Y_test_reg=train_test_split(X_cla,Y_cla,Y_reg,test_size=0.25,random_state=42)

X_train_reg=X_train_cla.drop(columns=["Age"])
X_test_reg=X_test_cla.drop(columns=["Age"]) #전체에서 빼지 말고 학습된 일부에서 빼도록 하자
#전체 행과 열의 수를 맞추기 위해서 일부에서 빼야 함

#studying!
doctor_cla=make_pipeline(
    StandardScaler(),
    RandomForestClassifier()
)

doctor_reg=make_pipeline(
    StandardScaler(),
    RandomForestRegressor()
)

doctor_cla.fit(X_train_cla,Y_train_cla)
doctor_reg.fit(X_train_reg,Y_train_reg)

pred_x=doctor_cla.predict(X_test_cla)
acc=accuracy_score(Y_test_cla,pred_x)

print(f'Model Accuracy: {acc*100:.2f}%')
print(
    """Complete model studying
    본 EMR은 XX대학교 "~" 수업 시간 때 제공받은 병원 데이터를 가져온 것이므로 
    무단으로 유포하지 않을 것을 선서합니다.
    사용자는 EMR을 함부로 유포하면 국내 의료법에 의거하여 법적 처벌을 받을 수 있다는 것을 인지합니다."""
    )

while True:
    def UserInfoGet(prompt,minVal,maxVal,Fl0at=False):
        try:
            while True:
                Uvalue= float(input(prompt)) if Fl0at else int(input(prompt))
                if minVal <= Uvalue <= maxVal:
                    return(Uvalue)
                print('Over range! Check ur value!')
        except Exception as ex:
            print(f'ERROR:{ex}')
    UserInfo={
        "Age":[UserInfoGet('Enter ur age',0,100)],
        "SBP":[UserInfoGet('Enter ur systolic blood pressure',30,300,Fl0at=True)],
        "DBP":[UserInfoGet('Enter ur diatolic blood pressure',30,200,Fl0at=True)],
        "BMI":[UserInfoGet('Enter ur bmi',10,50,Fl0at=True)],
        "GLU":[UserInfoGet('Enter ur glucose',10,400,Fl0at=True)]
    }
    #Classifier
    new_cla=pd.DataFrame(UserInfo)
    pred_cla=doctor_cla.predict(new_cla)[0]
    proba_cla=doctor_cla.predict_proba(new_cla)[0][1]
    #Regressor
    new_reg=new_cla.drop(columns=["Age"])
    pred_reg=doctor_reg.predict(new_reg)[0] #<-predict age

    if pred_cla == 0:
        print('User status: normal')
        print(f'Negative predict probability: {1-proba_cla*100:.2f}%')
        print(f'User age: {pred_reg}old')
    elif pred_cla == 1:
        print('User status: danger')
        print(f'Positive predict probability: {proba_cla*100:.2f}%')
        print(f'User age: {pred_reg:.0f}old')
    while True: 
        YesNo=input('retry? y|n:')
        if YesNo == 'y':
            break
        elif YesNo == 'n':
            break
        else:
            print('Type Error')
    if YesNo == 'y':
        print('continue!')
    elif YesNo == 'n':
        print('end!')
        break


#HbA1c(당화혈색소): 수명이 2~3개월인 적혈구에 설탕이 끈적하게 달라붙는데(당화), 한 번 붙은 설탕은 적혈구의 수명이 다 닳을 때까지 붙어있는다
#전체 적혈구 중에서 설탕이 묻어있는 적혈구의 '비율'을 나타내는 모델이다.
