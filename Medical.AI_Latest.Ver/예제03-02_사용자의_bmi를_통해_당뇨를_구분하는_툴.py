import numpy as np
from sklearn.linear_model import LogisticRegression

#환자 정보
X=np.array([
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

#앞에 배열 선언 
y=np.array([1,0,0,0,1,0,1,1,1,0])

#학습할 자칭 의사 틀 만들기
Doctor=LogisticRegression()
Doctor.fit(X,y)

print('모델 학습 완료')
print()

#bmi구간
print('bmi 측정')

weight=float(input('사용자의 몸무게를 입력 (자세히 입력)'))
height=float(input('사용자의 키를 입력 (자세히 입력)'))
height_m=height/100#m 변환

user_bmi_value=weight/(height_m**2)

while True:
    age=int(input('나이 입력: (0세부터 120세까지 기입 가능)'))
    if 0<age<120:
        print('올바른 값이 기입 됐습니다.')
        break
    else:
        print('제대로 된 값을 기입해 주세요')

bmi_result=round(user_bmi_value, 2)#round 함수 선언하여 소수점 2자리 만들기

#systolic,diatolic_bloodpresssure구간
systolic_bp=float(input('수축기 혈압 값을 입력'))
diatolic_bp=float(input('이완기 혈압 값을 입력'))

#함수선언
def classtify_bp(systolic_bp,diatolic_bp):
    if systolic_bp<=120 and diatolic_bp<=80:
        return('혈압 정상 단계!')
    elif systolic_bp<=129 and diatolic_bp<=89:
        return('고혈압 의심 1단계!')
    elif systolic_bp<=139 and diatolic_bp<=99:
        return('고혈압 2단계!')
    elif systolic_bp<=140 and diatolic_bp<=119:
        return('고혈압 3단계! 의사랑 상담 필요!')
    else:
        return('위기 단계! 즉시 입원!')

bp_result=classtify_bp(systolic_bp,diatolic_bp)

#당뇨 예측 가능성 설계
bmi_user_data=np.array([[age,bmi_result]])#2차원 배열
pred1=Doctor.predict(bmi_user_data)[0]#학습된 틀을 기반으로 유저 데이터 당뇨인지 아닌지 예측
proba1=Doctor.predict_proba(bmi_user_data)[0][1]# 유저 데이터 당뇨 예측 가능성 

#print(f'환자 정보:{bmi_user_data}') 배열 상태에서 값이 어떻게 나올지 궁금해서 써 봄

print(f'환자 정보/ 나이:{age} bmi 값:{bmi_result} 혈압 단계: 아래 참고')

if bmi_result<18.5:
    print('환자의 bmi 정보:{0:.2f} 저체중 범위입니다.'.format(bmi_result))
elif bmi_result<23:
    print('환자의 bmi 정보:{0:.2f} 정상 범위입니다.'.format(bmi_result))
elif bmi_result<25:
    print('환자의 bmi 정보:{0:.2f} 과체중 범위입니다.'.format(bmi_result))
elif bmi_result<30:
    print('환자의 bmi 정보:{0:.2f} 비만 범위입니다.'.format(bmi_result))
else:
    print('환자의 bmi 정보:{0:.2f} 비정상 범위입니다.'.format(bmi_result))

if pred1 == 0:
    print(f'당신은 정상이며 {bp_result} 단계 입니다.')
else:
    print(f'당신은 비정상이며 {bp_result} 단계 입니다.')
    print('그러므로 당뇨일 가능성을 보여드립니다. 결과:{0:.6f}'.format(proba1)) # X,y값의 환자 데이터가 부족하기에 소수점 6자리까지 부여. 연습용
