#본 EMR데이터는 XX대학교  극히 일부분 '틀'(내용X)을 가지고 진행했음을 알립니다
# 절대 실제 환자의 기록을 토대로 학습을 하지 않았으며 이를 가지고 유포하거나 타인에게 공유할 시, 국내 의료법에 위반된다는 걸 인지하고 있습니다.
#%%
import seaborn as sns
#seaborn: matplotlib 기반 업그레이드 통계 특화 버전(시각적 자료 퀄UP)
import matplotlib.pyplot as plt
import pandas as pd
import os
import pyreadstat as pt
from sklearn.preprocessing import LabelEncoder
#LabelEncoder: 문자의 형식을 숫자로 변환해주는 역할, 모델은 '숫자'만 읽을 수 있기에 변환을 해줘야 한다.(전처리 과정)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier #예측 판독기
from sklearn.ensemble import VotingClassifier #모델 평균 값 합치기 
from sklearn.impute import SimpleImputer #결측치 처리
from sklearn.metrics import accuracy_score, f1_score
#f1_score(정밀도):
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#sav,csv (sav->csv를 한다는 가정하에)|(암기 목적으로 직접 다 타이핑)
sav_file = 'Spssp[sav]training.sav'
csv_file = 'SpssTraining[csv]/training.csv' #확장자를 바꾸기 -> to_csv를 통해 같이 바뀜

if not os.path.exists(csv_file):
    d_sav,_= pt.read_sav(sav_file)
    print('Converting...')
    d_sav.to_csv(csv_file,index=False,encoding='utf-8-sig')
    print('Complete converting!')

#metadata
try:
    _,meta = pt.read_sav(sav_file,metadataonly=True)
    for c_name,c_label in meta.column_names_to_labels.items():
        label=c_label if c_label else ''Not Explaination'
        print(f'Explaination: {label}')
        print('Complete load!')
except Exception as e:
    print(f'ERROR: {e}')

#csv
csv_load = pd.read_csv('SpssTraining[csv]/training.csv') #csv 파일 안에

#굳이 LabelEncoder와 get_dummies를 쓰기(모르는 개념이니 공부)
# #해당 sav파일은 인코딩이 되어있는 상태지만 학습을 위해 바꿔보자
# def make_bmi(bmi):
#     if bmi < 18.5:
#         return '저체중'
#     elif bmi < 25.0:
#         return '정상'
#     else:
#         return '비만'

# #BMI열에 BMI_글자라는 열을 함수(make_bmi)값을 적용해 원본 sav파일의 새로운 열로 자리를 잡음
# csv_load['BMI_글자'] = csv_load['BMI'].apply(make_bmi)
# #번호 부여과정
# le=LabelEncoder()
# #'BMI_라벨'이라는 새로운 열로 인코딩한 결과들을 담아주기
# csv_load['BMI_라벨']=le.fit_transform(csv_load['BMI_글자'])
# #이렇게 되면 저체중:0,정상:1,비만:2가 '라벨'열에 들어가게 됨

# #pd.get_dummies 원~핫 코딩(열 안에 있는 내용물을 '열'형태로 찢게 하기)
# csv_load=pd.get_dummies(csv_load, columns=['BMI_글자'])
# #이렇게 되면 원본 파일을 담고 있는 csv_load에 'BMI_글자'가 글자_저체중,_정상,_비만 형태로 찢어지게 된다
# # 여기서 사용자 정보 기반(예:사용자는 저체중)으로 모델을 돌리면 _저체중:1,_정상:0,_비만:0으로 열 형태가 유지가 된다.  

#encoding:모델은 무조건 숫자만 읽는다 그러니 형식을 (문자->숫자)로 바꿔줘야 한다
# (0|1)<-이진 분류(astype(int))로 나타내기
pos_diabetes = (csv_load['HbA'] >= 6.5)
csv_load['diabetes_target'] = pos_diabetes.astype(int) # True: 1, False: 0 규칙 만들기(추가로 새로운 열도 만들기)

#seaborn
sns.countplot(x='diabetes_target', data=csv_load)
#countplot: seaborn내장어 기능 중 하나, 종류별로 묶어서(ex:0,1)개수를 세서 막대그래프로 표현
plt.show()


# %%

#26.4.17
