#본 EMR데이터는 XX대학교  극히 일부분 '틀'(내용X)을 가지고 진행했음을 알립니다
# 절대 실제 환자의 기록을 토대로 학습을 하지 않았으며 이를 가지고 유포하거나 타인에게 공유할 시, 국내 의료법에 위반된다는 걸 인지하고 있습니다.
################################################################################################################################################
#대사증후군을 3단계(0:정상,1:위험,2:고위험)로 나눠서 할 것|신경망 구성은 입력(6(카테고리 갯수에 맞춰서),24(hidden 전))
#은닉에선 24,12(사공이 많으면 배가 산으로 가듯이 최소 필요한 정보만 남겨둔다)|출력에선 12,3(마지막은 결과 값을 정상,위험,고위험으로 나타내기 위해서 3개로 설정)
#이번 주제는 DLvsML이며 어느 성능이 훨씬 우수한 지 확인을 하는 실험(정형 데이터를 기준으로 함)
import pandas as pd
import pyreadstat 
import os
import numpy as np
#DL
import torch
import torch.nn as nn 
import torch.optim as optim
#ML
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
#sav,csv
sav_file='Spss_sav(DL)/BP_Stat_Final_ExerData.sav'
csv_file='csv(DL)/ML,DL,test_file.csv'
#convert
try:
    if not os.path.exists(csv_file):
        sav_load,_=pyreadstat.read_sav(sav_file)
        print('Converting...')
        sav_load.to_csv(csv_file,index=False)
        print('Converted sav->csv')
except Exception as e:
    print(f'ERROR: {e}')
#csv
csv_load=pd.read_csv(csv_file)
#risk_count
risk_count=(
    (csv_load['SBP']>=130).astype(int)+
    (csv_load['GLU']>=126).astype(int)+
    (csv_load['BMI']>=25).astype(int)
    )
csv_load['Health']=np.where(risk_count==0,0,np.where(risk_count==1,1,2))
#풀어서 쓸 시
# if risk_count==0:
#     return 0
# elif risk_count==1:
#     return 1
# else:
#     return
#train,test
X=csv_load[['Gender','Age','BMI','WC','SBP','GLU']]
Y=csv_load['Health']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
#preprocessing
gen_div=['Gender']
other_div=['Age','BMI','WC','SBP','GLU']
gen_col=Pipeline([
    ('gen_simp',SimpleImputer(strategy='most_frequent'))
])
other_col=Pipeline([
    ('simp',SimpleImputer(strategy='median')),
    ('stad',StandardScaler())
])
total_col=ColumnTransformer(
    transformers=[
        ('gen_total',gen_col,gen_div),
        ('other_total',other_col,other_div)
    ])
#tuning
doctor=Pipeline([
    ('total',total_col),
    ('rf',RandomForestClassifier(random_state=42))
])
#GridSearchCV
param_rf={
    'rf__n_estimators':[10,50,100,200],
    'rf__max_depth':[3,5,7,10]
}
grid_rf=GridSearchCV(
    estimator=doctor,
    param_grid=param_rf,
    scoring='accuracy',
    cv=7,
    n_jobs=-1
)
#study and score
grid_rf.fit(X_train,Y_train)
pred_rf=grid_rf.predict(X_test)#에측
print(f'Machine Learning Model Accuracy: {accuracy_score(Y_test,pred_rf)*100:.2f}%')
#DL part - nn
class DLmodule(nn.Module):
    def __init__(self):
        super().__init__()
        #input
        self.input_layer=nn.Linear(6,24)
        self.input_relu=nn.ReLU()
        #hidden
        self.hidden_layer=nn.Linear(24,12)
        self.hidden_relu=nn.ReLU()
        #output
        self.output_layer=nn.Linear(12,3)
        self.sigmoid=nn.Sigmoid()
    def forward(self,ex_data):#길잡이
        x=self.input_layer(ex_data)
        x=self.input_relu(x)
        x=self.hidden_layer(x)
        x=self.hidden_relu(x)
        x=self.output_layer(x) 
        return x#교차엔트로피손실 채점 방식을 쓸 거기에 0~1로 변환해주는 시그모이드는 제외
#standard
scaler=StandardScaler()
X_scaler_tra=scaler.fit_transform(X_train)
X_scaler_test=scaler.transform(X_test) #데이터 스케일링 무조건 해줘야함!!!!!
#tensor->Y는 인덱스로 포장이 되었기에 순수 숫자 형태로 바꿔야함
#mps test
print('-'*50)
if torch.backends.mps.is_available():
    mps_t=torch.device('mps')
    print('GPU accessed')
else:
    print('CPU accessed')
#tensor로 담아주기
X_ten=torch.tensor(X_scaler_tra,dtype=torch.float).to(mps_t)
X_ten_t=torch.tensor(X_scaler_test,dtype=torch.float).to(mps_t)
#values:pandas기능 중 하나이며, 인덱스를 벗겨내고 깔끔한 내용물만 남도록 함->순수 숫자
#long:숫자를 900경까지 나타낼 수 있음(int64=long)
Y_ten=torch.tensor(Y_train.values,dtype=torch.long).to(mps_t)
Y_ten_t=torch.tensor(Y_test.values,dtype=torch.long).to(mps_t)
#setting
doctor_dl=DLmodule().to(mps_t)
#CrossEntropyLoss:BCE가 이진 분류면, 교차엔트로피손실은 다중 분류 버전 손실 함수이다. 선택지가 2개면 이진BCE
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(doctor_dl.parameters(),lr=0.01)
#test
for epochs in range(5000):
    prediction=doctor_dl(X_ten)
    loss=criterion(prediction,Y_ten)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
#DL accuracy score
#with:특정 구역을 만들어주는 기능,구역 안에서 사용자가 입력한 코드를 실행할 수 있음->끝나면 다시 원래 코드로 복귀
#no_grad:기울기 계산 금지,실전에선 오답 체크를 할 필요 없이 채점만 빠르게 하면 된다(연산 속도를 올림)
with torch.no_grad():
    dl_raw_output=doctor_dl(X_ten_t)#DL의사로 순수 값을 계산
    dl_pred=torch.argmax(dl_raw_output,dim=1)
    #argmax:가장 큰 값이 존재하는 공간(index)를 불러오기
    #dim(Dimension)=AI가 데이터를 읽는 시선의 방향을 의미함(자세한 설명은 아래)
    print(f'Deep Learning Model Accuracy: {accuracy_score(Y_ten_t.cpu(),dl_pred.cpu())*100:.2f}%')
    #accuracy_score는 cpu에서만 확인할 수 있기에 정보를 잠깐 .cpu()로 옮기기
#result
print('-'*50)
print('(ML vs DL) Result (Standard:0)')
#index셋팅
target_idx=0
real_answer=Y_ten_t[target_idx].item()#진짜 정답 출력
#ML pred_proba
#iloc:위에서부터 몇 번째 줄인지 숫자로 가져오는 기능
single_x_ml=X_test.iloc[[target_idx]]
prob_ml=grid_rf.predict_proba(single_x_ml)[0]#첫번째 사람 (0,1,2)확률구하기
#DL pred proba
#reshape:다시 형태를 잡아주는 기능(아래는 1차원->2차원으로 바꾸는 과정)-1은 자동으로 데이터 개수만큼 채우는 기능
single_x_dl=X_scaler_test[target_idx].reshape(1,-1)
single_x_dl_tensor=torch.tensor(single_x_dl,dtype=torch.float).to(mps_t)#mps로 보내기
#functional(함수)|softmax:[2.4,-1.2]이렇게 원시 점수가 출력되기에 무조건 확률(0~100%)로 바꿔줘야함-softmax->[정교한 값,정교한 값] 
#detach:데이터가 지나가는 모든 길,연산 기록 등을 남겨두는 기록을 분리하는 역할,순수 숫자만 남기고 싶을 때 사용
#doctor_dl
doctor_dl_x=doctor_dl(single_x_dl_tensor)
prob_dl=torch.nn.functional.softmax(doctor_dl_x,dim=1)[0].detach().cpu().numpy()
#[0]은 참고로 다시 순수 숫자만 남겨두기 위해서 리스트를 벗기는 역할
#결과 출력을 위해 다시 cpu로 변환,순수 숫자(numpy)로 남겨두기
print(f'Real answer: {real_answer}idx')
print('RF Result')
print(f'Normal(0): {prob_ml[0]*100:.1f}%| Danger(1): {prob_ml[1]*100:.1f}%| High Risk(2): {prob_ml[2]*100:.1f}%')
print('DL Result')
print(f'Normal(0): {prob_dl[0]*100:.1f}%| Danger(1): {prob_dl[1]*100:.1f}%| High Risk(2): {prob_dl[2]*100:.1f}%')

#dim(Dimension):데이터를 읽을 때 시선의 방향을 의미한다 예를 들어 dim값이 0일 경우 컴퓨터는 가장 바깥쪽 괄호(리스트 형태)로 취급함->위(열)에서 EMR의 모든 정보들을 참고
#그러면 각 환자를 구하는 과정에서 결과는 [환자1,환자2,환자4]<-이런 식으로 0,1,2에서 가장 값이 높은 애들을 출력하게됨(매우 잘못됨)
#이를 해결하기 위해 dim=1로 시선의 방향을 바꿔 행을 기준으로 보면 된다->1번에선 0,1,2중 가장 높은 값인 1:80|2번에선 0,1,2중 가장 높은 값인 0:50 -> [1,0] 
#각 환자의 최고치 정보가 담긴 리스트가 출력되는걸 확인할 수 있음