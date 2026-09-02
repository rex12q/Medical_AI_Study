#예제 6-1 (svm(Support Vector Machine), SVC (Support Vector Classifier))
import numpy as np
from sklearn.svm import SVC

#left:math, right:english
ma_en_score=np.array([
    [90.0,50.0],
    [40.5,90.1],
    [83.0,88.2],
    [44.5,49.3],
    [76.0,51.2],
    [59.0,58.1],
])

y=np.array([1,0,1,0,1,1])

teacher=SVC(probability=True) # 확률도 보고 싶으면 probability=True 추가
teacher.fit(ma_en_score,y)

print('선생님이 기준을 매김 (각 과목들은 50점 이상이면 합격)')

while True:
    user_age=int(input('나이:'))
    if 20<=user_age<=50:
        print('범위 내에 값이 기입됨')
        break
    else: 
        print('이 시험은 20세부터 50세까지 응시할 수 있는 시험입니다. 다시 쓰시오.')

while True:
    user_score1=float(input('수학 점수:'))
    if 0<= user_score1 <= 100:
        print('범위 내에 값이 기입됨')
        break
    else:
        print('디시 입력/ 범위(0~100)')

while True:
    user_score2=float(input('영어 점수:'))
    if 0<= user_score2 <= 100:
        print('범위 내에 값이 기입됨')
        break
    else:
        print('디시 입력/ 범위(0~100)')

#예측과 합격일 가능성
new_student_data=np.array([[user_score1,user_score2]])
new_student_pred=teacher.predict(new_student_data)[0] #user_result, negative or positive
#SVM은 기본적으로 선을 긋는 것에 집중하기 때문에, 확률을 계산하는 기능은 기본적으로 꺼져있다. 그러기 위해 SVC() 괄호 안에 probability=True 추가
new_student_proba=teacher.predict_proba(new_student_data)[0][1] #user_value+positive proba

#결과
def classify_score(new_student_pred):
    if new_student_pred == 1:
        return('합격!')
    else:
        return('불합격!')

user_result_score=classify_score(new_student_pred)

if new_student_pred == 1:
    print(f'학생의 정보| 나이:{user_age}, 수학 점수:{user_score1}, 영어 점수:{user_score2}, 합격 여부:{user_result_score}')
    print(f'합격자가 커트라인에 걸칠 수 있었던 확률{new_student_proba*100:.2f}')
else:
    print(f'학생의 정보| 나이:{user_age}, 수학 점수:{user_score1}, 영어 점수:{user_score2}, 합격 여부:{user_result_score}')
    print('좀 더 노력하세요 ㅈ만한 새끼야 :D')
