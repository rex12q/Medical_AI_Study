import numpy as np
#LogisticRegression(회귀)는 체온이 높아질 수록 발열 확률이 높아지는 '선'을 그려서 계산 (linear.model)
#그러나 DecisionTreeClassifier는 예/아니오 같은 '질문지'(Tree)를 만든다 (Tree)
#설명 가능한 AI라고 보면 된다->두통이나 기침 여부가 있을 경우 (ex)'체온 37도 이상이고 기침(두통 또는 두 조건 모두)을(를) 하는가?'
from sklearn.tree import DecisionTreeClassifier

#혈액 검사 수치: [HGB, HCT, RBC]
X = np.array([
    [14.0, 42, 4.8],
    [13.5, 40, 4.6],
    [11.0, 33, 3.9],
    [10.5, 31, 3.7],
    [15.0, 45, 5.1],
    [12.0, 36, 4.2],
])

y = np.array([0,0,1,1,0,1])  # 1=빈혈 의심(가짜), 0=정상(가짜)

new_doctor=DecisionTreeClassifier() # 틀 만들어주기
new_doctor.fit(X,y) #fit 학습은 회귀랑 동일

print('모델 학습 완료')

user_hgb=float(input('hgb(헤모글로빈) 입력: (예: 13.2)'))
user_hct=float(input('hct(헤마토크릿) 입력: (에: 40)'))
user_rbc=float(input('rbc(적혈구 수) 입력: (예: 4.5(갯수가 많기에 소수점 부여))'))

#어쨋든 머신러닝 모델 중 하나기 때문에 2차원 배열로 나타냄
new_data=np.array([[user_hgb,user_hct,user_rbc]])
new_pred=new_doctor.predict(new_data)[0] # 사용자의 음성,양성 에측 
#양성일 예측 가능성(추가로)
new_pred_proba=new_doctor.predict_proba(new_data)[0][1] #0:사용자, 1:양성, predict_proba:예측 가능성

#음성,양성을 판단해줄 문구 
if new_pred == 0:
    print('빈혈 의심 결과: 정상')
else:
    print('빈혈 의심 결과: 비정상')
    print(f'빈혈 의심 결과가 양성이기에 {new_pred_proba*100:.2f}% 결과가 나옴')
    

#HGB,HCT,RBC는 적혈구와 관련된 필수 혈액 검사 항목으로, 주로 빈혈 및 혈액 건강 상태를 평가
#RBC(Red Blood Cell, 적혈구 수): 산소를 운반하는 세포의 총 개수.
#HGB(Hemoglobin, 헤모글로빈/혈색소): 적혈구 내에서 실제 산소를 나르는 붉은 단백질 농도, 빈혈의 가장 핵심적인 지표.
#HCT(Hematocrit, 헤마토크릿/적혈구 용적률): 전체 혈액량 중 적혈구가 차지하는 백분율
