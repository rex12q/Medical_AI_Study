import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
from sklearn.linear_model import LogisticRegression

mpl.rcParams['font.family'] = 'AppleGothic'
mpl.rcParams['axes.unicode_minus'] = False

print('당뇨 예측 모델/실제 의료용 데이터')

X=np.array ([
    [50, 33.6],   # 환자1
    [31, 26.6],   # 환자2
    [32, 23.3],   # 환자3
    [21, 28.1],   # 환자4
    [33, 43.1],   # 환자5
    [30, 25.6],   # 환자6
    [26, 31.0],   # 환자7
    [29, 35.3],   # 환자8
    [45, 30.1],   # 환자9
    [41, 27.8]    # 환자10
])

y=np.array([1,0,0,0,1,0,1,1,1,1,])

bmi_model=LogisticRegression()
bmi_model.fit(X,y)

print('모델 학습이 완료 됐습니다.')

#weight:가중치 추출/coefficient: 계수
coef=bmi_model.coef_[0] #bmi_model의 논리회귀를 통한 깨달음(_로 표현) 그 후 coef(계수)를 이용하여 [나이,bmi]의 가중치를 가져옴

weight_age=coef[0] #여기에 나이를
weight_bmi=coef[1] #여기엔 bmi를 (왜냐하면 사용자가 [나이,bmi]순으로 짰기 때문)

#magnitude:절댓값으로 비교하기
mag_age=abs(weight_age)
mag_bmi=abs(weight_bmi) #abs()절댓값

print(f'해당 모델의 나이 가중치: {mag_age:.4f}%') #가중치 확률 나타내기
print(f'해당 모델의 bmi 가중치: {mag_bmi:.4f}%')

if mag_age>mag_bmi:
    print('이 모델은 "나이"를 중점으로 학습 됐습니다.')
elif mag_age<mag_bmi:
    print('이 모델은 "bmi"를 중점으로 학습 됐습니다.')
else:
    print('가중치 값이 동일합니다.')

#나이와 bmi를 나누기 
g_age=X[:,0] #0번째 열의 값을 가져와라
g_bmi=X[:,1] #1번쨰 열의 값을 가져와라 (0:age,1:bmi)

age=int(input('나이를 입력하세요'))
bmi=float(input('bmi를 입력하세요'))

new_data=np.array([[age,bmi]])
predict1=bmi_model.predict(new_data)[0]
proba1=bmi_model.predict_proba(new_data)[0][1]

print(new_data)
print(f'사용자 당뇨 여부 (0:없음 1:있음): {predict1}')

if predict1 == 1:
    print(f'사용자 당뇨 여부 가능성: {proba1*100:.2f}%')
else:
    print('사용자 당뇨 여부는 없습니다.')
    print(f'사용자 당뇨 여부 가능성: {proba1*100:.2f}%')

plt.scatter(g_age,g_bmi,color='blue',marker='o', label='기존 환자') #전체 데이터 값을 그래프로 표현
plt.scatter(age,bmi,color='red',marker='*', label='사용자')
plt.xlabel('AGE')
plt.ylabel('BMI')
plt.title('사용자의 당뇨 나타내기')
plt.legend() #label로 설정한 이름들을 화면에 표시
plt.show()

#

#구조
#numpy np 선언, sklearn.linear_model로 0과1 구분, LogisticRegression으로 논리회귀 생성
#배열(array): X=(나이,bmi지수),y=(참/거짓) <-학습할 데이터 (이래야 모델이 학습을 마친 후 분류를 할 수 있음)
#틀 만들어주기 (ex):bmi_model=LogsticRegression() -> 학습하기 bmi_model.fit(X,y)
#모델 분류(생성)완료!
#사용자 나이와 bmi받기 -> int(input('나이 입력'))/int(input('bmi입력'))
#사용자 나이와 bmi를 배열로 나타내기(LogsticRegression 2차원 배열) (ex)new_data=np.array([[age,bmi]])
#1.사용자는 음성(0),양성(1)인가 -> (ex)predict1=model.predict(new_data)[0] <-0
