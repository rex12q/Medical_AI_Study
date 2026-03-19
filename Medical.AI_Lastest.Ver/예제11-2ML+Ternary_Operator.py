import numpy as np
import pandas as pd
from sklearn.svm import SVC #0/1 같은 클래스로 나누는 모델
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline #의사 대가리 구성할 파이프라인 
from sklearn.preprocessing import StandardScaler #모든 값에 가중치를 동일하게 둘 공평한 툴(같은 체급으로)
from sklearn.model_selection import train_test_split #문제,정답지와 시험,정답지 분배

#랜덤으로 돌릴 숫자 고정
np.random.seed(42)
p_samples=300

#범위 지정
age_samples=np.random.randint(0,100,p_samples)
bmi_samples=np.random.uniform(17.5,40,p_samples)
sysbp_samples=np.random.randint(90,220,p_samples)
#위험요소 누적시스템
risk=[] #리스트 안에 차곡 모은 이 결과를 학습해야함
for i in range(p_samples):
    if bmi_samples[i] >= 25.5 and sysbp_samples[i] >= 140: #i번째 bmi환자, i번째 수축기혈압 환자
        risk.append(1)
    else:
        risk.append(0)
#데이터셋
db_df=pd.DataFrame({
    "age":age_samples,
    "bmi":bmi_samples,
    "sysbp":sysbp_samples,
    "risk":risk
})

#분배(train)
X=db_df[['age','bmi','sysbp']] #2차원 배열 
Y=db_df['risk']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=42)
#의사양반 
doctor=make_pipeline( #매끈매끈하다 매끈매끈한 파이프라인으로 연결
    StandardScaler(), #UFC 체급 맞추기 
    SVC(probability=True)
)
doctor.fit(X_train,Y_train) #계속 틀림(의사양반은 기출연습문제과 기출연습정답지를 보고 학습을 한다)
#정확도 
acc_pred=doctor.predict(X_test) #실제 시험 문제로 본인 실력 확인(예열)
acc_data=accuracy_score(Y_test,acc_pred) # 실제 시험 정답지와 본인이 풀었던 문제를 비교해 본인의 정확한 실력을 파악 

print(f'의사양반의 정확도: {acc_data*100:.2f}%')
print('의사양반은 준비가 됐습니다.')

#사용자 정보
while True: 
    def get_value(prompt,min_val,max_val,is_float=False): #함수로 받아줄 구조를 설계한다 Ternary Operator(삼항연산자)
        #보다시피 함수에는 is_float=False라는 기본값을 설정했다
        while True: 
            try: #try~except 패턴
                u_value = float(input(prompt)) if is_float else int(input(prompt))
                #A를 해라 만약 조건이 is_float(True상태)이라면. 아니면 B(False상태)를 해라
                #함수 틀을 보면 is_float=False로 되어있기에 '기본값'은 정수가 된다. 
                #is_float을 보면 '내장어'가 아닌 변수명에 불과하다. 그러나 if~else절을 통해 컴퓨터와 사용자는 참일 때 float, 거짓일 때 int를 출력하는 것을 알 수 있다.
                if min_val <= u_value <= max_val:
                    return u_value #while은 return문이 잘 받아서 def를 끝내버린다(break)
                print(f'{min_val}~{max_val}까지 입력할 수 있습니다.') #(return이 있기에 '나머지 모든 경우'에 해당됨)
            except ValueError: #ValueError는 type이 아예 다를 때 나타냄 (예외 시)
                print('해당 문자는 입력할 수 없어용!')
    user_age=get_value('나이 입력:',0,100) #함수 틀에 맞춰 is_float에 대한 값은 없기에 '기본값'이 적용된다
    user_bmi=get_value('BMI 입력:',17.5,40, is_float=True) #사용자 셋팅.
    user_sysbp =get_value('수축기 혈입:',90,220) #마찬가지로 값이 없기에 '기본값'이 적용된다.
    #사용자 데이터 
    new_data=pd.DataFrame([[user_age,user_bmi,user_sysbp]], columns=('age','bmi','sysbp')) #df는 무조건 열을 columns라고 지어야 함
    new_pred=doctor.predict(new_data)[0]
    new_proba=doctor.predict_proba(new_data)[0][1]
    #result
    print(f'사용자 정보| 나이:{user_age}, bmi:{user_bmi}, 수축기 혈압:{user_sysbp}')
    if new_pred == 0:
        print('음성임다')
        print(f'음성 확률: {(1-new_proba)*100:.2f}') #0과 1사이의 값을 추출하는 것이기에, 심지어 음성을 출력하기에 1를 빼야 음성일 확률을 나타낼 수 있음
    else:
        print('양성임다')
        print(f'양성 확률: {new_proba*100:.2f}')
    print('모든 결과가 다 나왔슴다')
    while True:
        user_again = input('테스트 재진행? (y/n)')
        if user_again == 'y':
            break
        elif user_again == 'n':
            break
        else:
            print('y/n만 입력!')
    if user_again == 'y':
        print('재개')
    elif user_again == 'n':
        print('종료')
        break




# while True:
#     while True:
#         user_age=int(input('나이 입력'))
#         if 0 <= user_age <= 100:
#             break
#         else:
#             print('올바른 값을 입력하세용')
#     while True:
#         user_bmi=float(input('bmi 입력'))
#         if 12.5 <= user_bmi <= 50:
#             break
#         else:
#             print('옵바른 값을 입력하세용')
#     while True:
#         user_sysbp=int(input('수축기 혈압 입력'))
#         if 90 <= user_sysbp <= 220:
#             break
#         else:
#             print('올바른 값을 입력하세용')