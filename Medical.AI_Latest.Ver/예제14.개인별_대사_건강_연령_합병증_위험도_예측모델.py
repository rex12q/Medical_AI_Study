import pandas as pd
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#ensemble:'조화'라는 개념에서 RandomForestClassifier:'많은 측정 모델' 개념이 나오다 
#정확도에 모든 것을 올인한 내재된 여러 개의 분류 성능을 볼 수 있다.
from sklearn.ensemble import RandomForestClassifier
#위와 동일하며, 분류와 다르게 정확한 숫자를 '예측'할 때 쓰는 도구이다 
#예시로 bmi와 혈당을 바탕으로 환자의 5년 뒤 예상 심혈관 나이를 소수점까지 정확하게 예측하는 개념이다
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
#카테고리가 같은 분석 방법들이 여러가지 있을 때 '합의'(여러가지 ml의 확률 중간을 찾기)를 할 수 있는 기능이다.
# 모델의 확률 차이가 너무 나면 굳이 안 쓰고 주석 처리를 해도 된다. 
from sklearn.ensemble import VotingClassifier

#데이터 불러오기
s_df=pd.read_csv('[csv]sample.csv')

#판정 결과가 없기에 직접 받아서 만들기 
s_risk=[]
for s in range(len(s_df)):
    if s_df["glucose"][s] >= 126 and s_df["bmi"][s] >= 25.0 and s_df["blood_pressure"][s] >= 130:
        #glucose는 126이상, bmi는 25.0이상, bp는 130이상(고혈압 1단계)이 양성 기준이다
        s_risk.append(1)
    else:
        s_risk.append(0)

#당뇨병 판정 이름 지어주기#check
s_df["diabetes_target"]=s_risk

#DF #check 
#d1_df=s_df.drop(columns=["patient","diabetes_target"]) check
#->이렇게 짜면 전체적으로 컬럼이 삭제되는 것이기 때문에 별도로 분리해서 쓰지 말고 한정된 공간에서 쓰기

#학습할 거 나눠주고 셋팅|(전체)
X=s_df.drop(columns=["patient","diabetes_target"]) #답을 모르고 학습해야 진짜 실력이 나온다
#patient는 애초에 비교 자체를 못하기에 그냥 버리고 간다 (전체 삭제)
#분류를 위한 학습과 예측을 위한 학습을 나누기 (분류 학습)
Y_cla=s_df["diabetes_target"] #0,1을 나타내는 당뇨 판정결과 (추후에 한 번 더 쓰인다 cla,svc)

#분류를 위한 학습과 예측을 위한 학습을 나누기 (예측 학습)
Y_reg=s_df["age"]

#분류학습과 예측학습을 위해 Y_train,test(cla,reg)추가
X_train,X_test,Y_train_cla,Y_test_cla,Y_train_reg,Y_test_reg=train_test_split(X,Y_cla,Y_reg,test_size=0.25,random_state=42)
#Y_train_cla는 X_train과 단짝

#나이는 따로 '예측' 학습,정답지를 만들어준다|(나이)
X_train_reg=X_train.drop(columns=["age"])
X_test_reg=X_test.drop(columns=["age"])#답을 모르고 학습해야 진짜 실력이 나온다
#Y_train_reg는 X_train_reg랑 단짝

#진료과1: 정확도를 끌어올릴 음/양성 판독기 (확률내기,분류)
doctor_cla=make_pipeline(
    StandardScaler(),
    RandomForestClassifier() #정확도를 끌어올림
)

#진료과2: 확실한 경계선 긋기(support vector), 복잡한 경계선 분석 전문 (확률내기,분류)
doctor_svc=make_pipeline(
    StandardScaler(),
    SVC(probability=True)
)

#진료과 1,2 합치기(둘 다 정확도 100이 떴지만 연습을 위해서..)
doctor_vote=VotingClassifier(
    estimators=[ #추정량: 평균,분산을 추정하는 함수
        ('cla',doctor_cla),
        ('svc',doctor_svc)
    ], #콤마 조심
    voting='soft' #'soft': 두 ml 확률의 평균을 내줌
)

#randomforestclassifier와 svc는 카테고리가 같기에 둘의 확률 결과가 너무 차이나면 주석처리
#그게 아니라 votingclassifier를 이용하여 확률을 합칠 것 

#진료과3: n세 질병 예측, 혈관, 나이 등 구체적 '수치'로 나타내기 전문 (회귀)
#주의할 점: y인 정답지 훈련이 아닌 '나이'와 같은 구체적 내용이 담긴 숫자 훈련이 가능함
doctor_reg=make_pipeline(
    StandardScaler(),
    RandomForestRegressor()
)

#check
doctor_vote.fit(X_train,Y_train_cla) # 두 ml을 합친 학습/당뇨 여부
doctor_reg.fit(X_train_reg,Y_train_reg) #나이 예측하는 학습 방법

#check 
# accpred_X=doctor_cla.predict(X_test)#분류로 예측
# accpred_Y=accuracy_score(Y_test,accpred_X)

accpred_X1=doctor_vote.predict(X_test)
accpred_Y=accuracy_score(accpred_X1,Y_test_cla) #정답지와 비교 

print(f'Model Accuracy:{accpred_Y*100:.2f}%')
if accpred_Y >= 0.50:
    print("This Model's accuracy is above average. Don't Worry :)")
else:
    print("This Model's accuracy is under average. Don't trust 100percent")

print("Complete Model Studying!")

#사용자 정보 받기
while True:
    def UserValue(prompt,min_val,max_val,is_float=False): #삼항연산자 기억을 잘 못하니 너가 직접 써보자
        while True:
            try:
                user_value=float(input(prompt)) if is_float else int(input(prompt))
                if min_val <= user_value <= max_val:
                    return(user_value)
                print('Over range! Again enter value')
            except ValueError:
                print('Type Error! Edit Type')
    UserInfo={
        "glucose":[UserValue('Enter your glucose',50,450)], #글루코스 
        "bmi":[UserValue('Enter your bmi',14.5,50.5),],
        "age":[UserValue('Enter your age',0,100)], 
        "blood_pressure":[UserValue('Enter your bloodpressure(systolic)',50,300)] 
        #수축기 이완은 0부터 120이하는 정상 140이상부터는 130부터는 고혈압 1단계에 포함된다
    }
    #분류 학습과 '예측' 학습 표 나누기
    #(분류 겸 전체 데이터)
    UserInfo_cla=pd.DataFrame(UserInfo)
    #(예측)
    UserInfo_reg=UserInfo_cla.drop(columns=["age"]) #cla에 age가 그대로 있기에 지워서 실행
    #모델이 학습한 것과 다른 개수이기에 빼야 함
    Upred_data=doctor_vote.predict(UserInfo_cla)[0]
    Uproba_data=doctor_vote.predict_proba(UserInfo_cla)[0][1]
    #사용자 예측 가능성 
    Ureg_data=doctor_reg.predict(UserInfo_reg)[0] #Regressor는 확률을 모른다 그러므로 predict_proba->predict
    if Upred_data == 0:
        print('User status: Normal')
        print(f'Negative Predict Probability{(1-Uproba_data)*100:.2f}%')
        print(f"Model predicts user's body age{Ureg_data:.0f}old")
    elif Upred_data == 1:
        print('User status: Danger')
        print(f'Positive Predict Probability{Uproba_data*100:.2f}%')
        print(f"Model predicts that user's body age{Ureg_data:.0f}old")
    else:
        print("Model can't measure user status ")
    while True:
        YesNo=input('Can you restart this test? (press y|n)')
        if YesNo == 'y':
            break
        elif YesNo == 'n':
            break
        else:
            print('You can only enter two spell(y|n)')
    if YesNo == 'y':
        print('Countinue test')
    elif YesNo == 'n':
        print('End test')
        break


