import numpy as np
import pandas as pd
#model_selection: 공정하게 테스트 할 수 있는 도구들의 상자
#train_test_split: 시험용과 공부할 문제를 나누는 작업. 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression #회귀 (학습할 틀) 
from sklearn.metrics import accuracy_score # 정확성 (학습한 모델과 결과가 유사한 지)

#():DF, {}: Dictionary, []: List
all_df=pd.DataFrame({
    "age":   [30,35,40,45,50,55,60,65], #나이 
    "bmi":   [20,22,24,26,28,30,32,27], #수치 
    "sysbp": [120,125,130,135,140,150,160,145], #수축기 혈압
    "risk":  [0,0,0,0,1,1,1,1],   # 1=위험(가짜)
})

#X=문제지/Y=정답지
X=all_df[['age','bmi','sysbp']] #여러 개의 열을 뽑아서 여전히 DF형태로 유지 ->DATA SET 2차원 배열
Y=all_df['risk'] #딱 하나의 열만 뽑아서 리스트 형태로 만들거임. -> 1차원 배열로 충분

#X_train: 모델이 공부할 때 '볼 문제지'(나이,bmi,혈압) 75%
# X_test: 모델이 공부를 마치고 바로 시험지를 풀 단계 (위험 여부) 25%
# Y_train: 모델이 공부를 '마친 후 정답 체크' 75%
# Y_test: 실제 문제를 푼 뒤 정답 체크 25% 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=42) 
#test_size=0.25는 전체 데이터 100개 중 25개는 시험용으로 따로 빼두겠다는 뜻
#42는 데이터를 섞을 때 사용하는 난수 번호(참고로 42는 개발자들 사이의 일종의 관습) 어떤 숫자를 넣든 숫자만 고정하면 언제 실행해도 똑같이 섞인 데이터를 얻을 수 있다.
#참고로 42번 레시피를 보고 데이터를 섞어줘 라고 명령하는 것이다. (메뉴얼이 다 있는 걸로 앎),(결과의 일관성)

#회귀 틀
model=LogisticRegression()
model.fit(X_train,Y_train)#'학습용 데이터'만 가지고 학습시키기(전체 데이터로 하면 안됨)->판정할 의사 만들어주기

#시험 데이터로 '예측'(평가)하기 
pred_test = model.predict(X_test)
#실제 답안이랑 pred_test가 학습한 결과를 비교해 정확성을 나타내기 acc
acc=accuracy_score(Y_test, pred_test)
print(f'모델의 정확도: {acc*100:.2f}')

#사용자 데이터 
u_age=int(input('나이 입력'))
u_bmi=float(input('bmi 입력'))
u_sysbp=int(input('수축기 혈압 입력'))

#사용자 데이터를 받아서 예측하고 예측 가능성 알아보기
#학습할 때는 이름표(column names)가 있는 pandas 데이터프레임으로 공부했지만, 예측할 때는 이름표 없는 생짜 numpy배열로 할 이유가 없음
#pandas는 열이름 정할 때 col같은 줄임말을 쓰면 안된다 -> columns
new_data=pd.DataFrame([[u_age,u_bmi,u_sysbp]], columns=['age', 'bmi', 'sysbp'])#ML은 항상 2차원 배열을 요구
new_pred=model.predict(new_data)[0]#사용자 데이터 인덱스 (음성,양성 예측)
new_proba=model.predict_proba(new_data)[0][1]#사용자 데이터 인덱스랑 사용자가 양성인 지

if new_pred == 0:
    print('사용자는 음성입니다.')
    print(f'양성 사용자가 걸릴 가능성: {new_proba*100:.2f}')
else:
    print('사용자는 양성입니다.')
    print(f'양성 사용자가 걸릴 가능성: {new_proba*100:.2f}')

#1.배열과 학습할 회귀틀, DF, 학습지와 정답지 비교를 위한 도구 쓰기 (numpy,LogisticRegression,pandas,accuracy_score,train_test_spilt)
#2.데이터 셋 만들어주기 (panda)
#3.만들어준 데이터 셋을 통해 모델이 학습할 데이터와 데이터 셋을 비교하는 과정을 구성(X_train,Y_train,X_test,Y_test), ()