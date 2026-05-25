# #GPU mps쓰기
# import torch
# data=[[112,344],[111,122]] #배열 형태
# t_data=torch.tensor(data)
# print(f'mps를 안 쓰고 그냥 담아준 텐서: {t_data}')
# if torch.backends.mps.is_available(): #mps를 쓸 수 있을 때
#     mps_model=torch.device('mps') #mps 선택하기
#     fast_model=t_data.to(mps_model) #GPU를 안 쓴 텐서를 쓸 수 있게 to이용
#     print(f'mps를 쓴 텐서: {fast_model}')
# else:
#     print('CPU로 전환 ')  #26.5.11
##########################################################################################################################################
# import torch 
# import torch.nn as nn #신경망
# import torch.optim as optim#최적화
# X=torch.tensor([[1.0],[2.0],[3.0]]) #공부 시간 
# Y=torch.tensor([[50.0],[60.0],[90.0]]) #점수
# brain_model=nn.Linear(in_features=1,out_features=1)#입출력 각 1ㄷ1
# criterion=nn.MSELoss()
# optimizer=optim.SGD(brain_model.parameters(),lr=0.01)
# #학습률은 0.01로 해서 0에 수렴할 수 있도록 미세조정을 함 브레인 모델에 매개변수를 이용해 오차를 줄임
# for training in range(100):
#     optimizer.zero_grad #찐 성능 확인 전에 내용 초기화
#     prediction=brain_model(X)#문제 풀기
#     loss=criterion(prediction,Y)#문제 푼 것과 정답지의 오차가 어디서 발생했는지 확인
#     loss.backward()#오차가 발생한 곳을 편미분으로 역추적(결과->은닉->입력)하여 0에 수렴할 수 있도록 도움
#     optimizer.step()#미분으로 결과 전체 값을 수정하는게 아닌,편미분을 이용해 각 값을 수정해 오차 줄임
#     #초기화->예측(공부)->오답체크->편미분 역추적->오답수정(0에 수렴할 수 있도록)
########################################################################################################################################
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# patients_data=torch.linspace(-5,5,100)
# try:
#     if torch.backends.mps.is_available():
#         mps_device=torch.device('mps')
#         fast_machine=patients_data.to(mps_device)
#         print('Converted GPU')
#     else:
#         print('Converted CPU')
# except Exception as e:
#     print(f'ERROR: {e}')
# #ReLU,Sigmoid
# relu_filter=nn.ReLU()
# clean_output=relu_filter(patients_data)
# sigmoid_filter=nn.Sigmoid()
# clean_probability=sigmoid_filter(patients_data)
# plt.figure(figsize=(14,6))
# plt.subplot(1,2,1)
# plt.plot(patients_data,clean_output.numpy(),color='red')
# plt.title('ReLU filter')
# plt.subplot(1,2,2)
# plt.plot(patients_data.numpy(),clean_probability.numpy())
# plt.title('Sigmoid filter')
# plt.show() #26.5.13
####################################################################################################################################
# import torch 
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt 
# x=torch.tensor([[1.0],[2.0],[3.0]])
# y=torch.tensor([[34.0],[50.0],[80.0]])
# brain=nn.Linear(in_features=1,out_features=1)# 점수,문제 각각 1
# loss_model=nn.MSELoss()
# optimizer=optim.SGD(brain.parameters(),lr=0.01)#확률적 경사 하강법,보폭설정,매개변수
# for d in range(1000):
#     brain.zero_grad # 기억지우기
#     prediction=brain(x)#시험
#     loss=loss_model(prediction,y)#MSE채점 결과에는 오차 존재
#     loss.backward()#편미분으로 오차가 발생한 값을 역추적
#     optimizer.step()#SGD가 적용된 변수에 step()으로 lr을 곱해 가중치를 수정->0에 수렴
# #result
# print(f'When you study 1 hour{brain(torch.tensor([[1.0]])).item():.2f}')
# print(f'When you study 2 hour{brain(torch.tensor([[2.0]])).item():.2f}')
# print(f'When you study 3 hour{brain(torch.tensor([[3.0]])).item():.2f}')
# #ex3
# patients_data=torch.linspace(-10,10,200)
# if torch.backends.mps.is_available():
#     gpu=torch.device('mps')
#     model=patients_data.to(gpu)
#     print('Converted GPU')
# else:
#     print('Converted CPU')
# relu=nn.ReLU()
# clean_data=relu(patients_data)
# sigmoid=nn.Sigmoid()
# clean_probability=sigmoid(patients_data)
# plt.figure(figsize=(14,6))
# plt.subplot(1,2,1)
# plt.plot(patients_data.numpy(),clean_data.numpy(),color='blue')
# plt.title('ReLU filter')
# plt.subplot(1,2,2)
# plt.plot(patients_data.numpy(),clean_probability.numpy(),color='red')
# plt.title('sigmoid filter')
# plt.show() 26.5.14
########################################################################################################################################
# import torch
# from torch.utils.data import DataLoader,Dataset#기능 상속
# class EMRdata(Dataset):#class로 묶음을 만들어서 Dataset의 기능을 상속
#     #self:'자신',위치를 자신으로 지정을 해주기
#     def __init__(self):#init:초기 작업,표를 만들 수도 있고 pd를 통해 파일을 불러올 수도 있는 기능
#         self.x=torch.randn(100,2)#행:100,열:2
#         self.y=torch.randint(0,2,(100,2))#0~2미만,0,1만 가능,
#     def __len__(self):#크기 알림
#         return len(self.x)
#     def __getitem__(self,idx):#idx(index):인덱스 번호로 정보 불러오기
#         return self.x[idx],self.y[idx]
# emr_data=EMRdata()
# #emr_data,DataLoader로 불러오기,cpu과부하로 인해 batch는 10으로,정보 암기 방지,섞기
# emr_load=DataLoader(emr_data,batch_size=10,shuffle=True)
# for batch_x,batch_y in emr_load:
#     print(f'cpu로 전달되는 정보량: {len(batch_x)}개')#len을 이용해서 정보의 
#     print(f'전달되는 정보 크기: {batch_x.shape}')#shape:튜플(수정 불가 정보),torch Size로 출력
# #26.5.18
##########################################################################################################################################
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# patients_data=torch.linspace(-5,5,20)
# if torch.backends.mps.is_available():
#     fast_model=torch.device('mps')
#     convert_mps=patients_data.to(fast_model)
#     print('test accessed! (Result: GPU)')
# else:
#     print('test accessed! (Result: CPU)')
# #ReLU
# relu=nn.ReLU().to(fast_model)
# relu_filter=relu(patients_data)
# #sigmoid
# sigmoid=nn.Sigmoid()
# sigmoid_filter=sigmoid(convert_mps)
# #graph
# plt.figure(figsize=(14,6))
# plt.subplot(1,2,1)
# plt.plot(patients_data.numpy(),relu_filter.numpy(),color='red')
# plt.title('relu filter')
# plt.subplot(1,2,2)
# plt.plot(patients_data.numpy(),sigmoid_filter.numpy(),color='blue')
# plt.title('subplot filter')
# plt.show() 26.5.19
########################################################################################################################################################################################
# import torch
# from torch.utils.data import DataLoader,Dataset
# class EMRData(Dataset):
#     def __init__(self): #초기 작업(랜덤 행렬표 생성)
#         self.patients_x=torch.randn(100,2)
#         self.patients_y=torch.randint(0,2,(100,1),dtype=torch.float)#음성,양성 들어갈 칸
#     def __len__(self):
#         return len(self.patients_x)
#     def __getitem__(self,idx):
#         return self.patients_x[idx],self.patients_y[idx]
# new_data=EMRData()
# new_load=DataLoader(new_data,batch_size=10,shuffle=True)
# #EMRData()기능을 가지고 정보를 무작위로 섞은 후 배치를 10씩 옮김
# for batch_x,batch_y in new_load:
#     print(f'{len(batch_x)}')
#     print(f'{batch_x.shape}')
#     break
############################################################################################################################################################################################################
# import torch
# import torch.nn as nn
# class DLmodule(nn.Module):
#     def __init__(self):#초기 작업
#         super().__init__()#부모(super),상위 클래스에서 모듈 받아오기
#         self.input_layer=nn.Linear(in_features=4,out_features=16)
#         self.input_relu=nn.ReLU()#변수명 다르게 설정
#         self.hidden_layer=nn.Linear(in_features=16,out_features=8)
#         self.hidden_relu=nn.ReLU()
#         self.output_layer=nn.Linear(in_features=8,out_features=1)
#         self.output_relu=nn.ReLU()
#         self.sigmoid=nn.Sigmoid()
#     def forward(self,user_data):#정보를 어떤 방향으로 안내할 것인가
#         x=self.input_layer(user_data)
#         x=self.input_relu(x)
#         x=self.hidden_layer(x)#user_data를 x에 삽입->x는 곧 사용자의 데이터를 의미
#         x=self.hidden_relu(x)
#         x=self.output_layer(x)#relu는 따로 쓰지 않음->0이하의 값들을 전부 drop하기에 결과가 안 나올 수도 있음
#         x_sigmoid=self.sigmoid(x)#sigmoid를 통해 0~1확률값으로 변환
#         return x_sigmoid
# doctor=DLmodule()
# def Info(prompt,minval,maxval,is_float=True):
#     while True:
#         try:
#             uvalue=float(input(prompt)) if is_float else int(input(prompt))
#             if minval <= uvalue <= maxval:
#                 return uvalue
#             print('fk off')
#         except Exception as e:
#             print(f'{e}')
# age=Info('몇 살임?',0,100,is_float=False)
# bmi=Info('bmi?',0,100)
# sysbp=Info('sysbp?',0,200)
# diabp=Info('diabp?',0,200)
# total=[age,bmi,sysbp,diabp]
# total_result=torch.tensor([total],dtype=torch.float)
# #2차원 배열 만들어주기,DL실수로 바꿔주기
# result=doctor(total_result)
# print(f'{result.item()*100.:.2f}')#포맷문에선 그냥 값을 가져올 수 없기에 꼭 item()써야함
# #26.5.19
###################################################################################################
# import torch
# import torch.nn as nn 
# import torch.optim as optim
# from sklearn.preprocessing import StandardScaler
# class DLmodule(nn.Module):#Module기본 셋팅 받아오기
#     def __init__(self): #initialize 초기 단계
#         super().__init__()#super 최상단 클래스 가져오기(콜론 안 써도 됨)
#         #input
#         self.input_layer=nn.Linear(in_features=4,out_features=16)
#         self.input_relu=nn.ReLU()
#         #hidden
#         self.hidden_layer=nn.Linear(in_features=16,out_features=8)
#         self.hidden_relu=nn.ReLU()
#         #output
#         self.output_layer=nn.Linear(in_features=8,out_features=1)
#         self.sigmoid=nn.Sigmoid()#0이하 값drop 이슈로 바로 시그모이드로 확률 변환
#     def forward(self,ex_data):#forward를 통해 ex_data의 길잡이 역할을 도와줌
#         #input
#         x=self.input_layer(ex_data)
#         x=self.input_relu(x)
#         #hidden
#         x=self.hidden_layer(x)
#         x=self.hidden_relu(x)
#         #output
#         x=self.output_layer(x)
#         self.sigmoid_x=self.sigmoid(x)
#         return self.sigmoid_x
# #데이터
# raw_X = [
#     [21, 23.0, 80,  85],  # 아주 건강한 사람
#     [45, 27.5, 135, 110], # 당뇨 위험군 사람
#     [60, 31.0, 150, 140], # 중증 당뇨 사람
#     [25, 21.0, 110, 90],  # 건강한 사람
#     [55, 29.0, 145, 125]  # 당뇨 사람
# ]
# #정답지
# raw_Y=[[0],[1],[1],[0],[1]]
# #표준화
# standard=StandardScaler()
# standard_x=standard.fit_transform(raw_X)#형식을 바꿔야함(fit_transform)
# #torch로 담고 x,y 나누기
# X_train=torch.tensor(standard_x,dtype=torch.float)
# Y_train=torch.tensor(raw_Y,dtype=torch.float)
# #doctor!
# doctor=DLmodule()
# #BCE:Binary Cross Entropy
# bce=nn.BCELoss()#이진 교차 엔트로피는 참일 경우 1에 수렴,거짓일 경우 0에 수렴하도록 채점이 설정됨
# #optimizer|Adam을 가지고 docto의 매개변수와 보폭 범위를 설정해 값이 최대한 0에 수렴할 수 있도록 '최적화' 작업 진행
# optimizer=optim.Adam(doctor.parameters(),lr=0.01)
# #학습->채점->초기화->오답체크(역추적)->업데이트 된 사항을 가지고 성능 테스트
# for epochs in range(1000):
#     prediction=doctor(X_train)
#     loss=bce(prediction,Y_train)
#     optimizer.zero_grad()
#     loss.backward()#채점한 부분에서 오류난 부분 편미분으로 역추적
#     optimizer.step()#업데이트 된 사항을 가지고 다시 진행
# #결과
# result=doctor(X_train)
# print('결과')
# print(f'{result[0].item()*100:.1f}')
# print(f'{result[1].item()*100:.1f}')
# print(f'{result[2].item()*100:.1f}')
#########################################################################################################################################################################################################################################################################################################
#ML
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
#DL
import torch
import torch.nn as nn
import torch.optim as optim
#csv
csv_load=pd.read_csv('Training.csv')
#risk_count
risk_count=(
    (csv_load['SBP']>=130).astype(int)+
    (csv_load['GLU']>=126).astype(int)+
    (csv_load['BMI']>=30).astype(int)
)
#positive_col
csv_load['positive_col']=np.where(risk_count==0,0,np.where(risk_count==1,1,2))
#train_test
x=csv_load[['Gender','Age','BMI','SBP','WC','GLU']]
y=csv_load['positive_col']
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#col
gen_col=['Gender']
other_col=['Age','BMI','SBP','WC','GLU']
g_pipe=Pipeline([
    ('g_impute',SimpleImputer(strategy='most_frequent')),
])
o_pipe=Pipeline([
    ('o_impute',SimpleImputer(strategy='median')),
    ('o_stad',StandardScaler())
])
total_pipe=ColumnTransformer(
    transformers=[
        ('total_g',g_pipe,gen_col),
        ('total_o',o_pipe,other_col)
    ]
)
#doctor
doctor_ml=Pipeline([
    ('total_p',total_pipe),
    ('rf',RandomForestClassifier(random_state=42))
])
#GridsearchCV
param_rf={
    'rf__n_estimators':[10,100,200,300],
    'rf__max_depth':[3,6,10]
}
tuning_rf=GridSearchCV(
    estimator=doctor_ml,
    param_grid=param_rf,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)
tuning_rf.fit(X_train,Y_train)
pred_rf=tuning_rf.predict(X_test)
print(f'Machine Leearing Accuracy: {accuracy_score(pred_rf,Y_test)*100:.1f}')
#DL
class EMRmodule(nn.Module):
    def __init__(self):
        super().__init__()
        #input,hidden,output
        self.input_layer=nn.Linear(6,24)#csv_load['']
        self.input_relu=nn.ReLU()
        self.hidden_layer=nn.Linear(24,12)
        self.hidden_relu=nn.ReLU()
        self.output_layer=nn.Linear(12,3)#0,1,2
        self.sigmoid=nn.Sigmoid()
    def forward(self,ex_data):#데이터 경로 설정
        x=self.input_layer(ex_data)
        x=self.input_relu(x)
        x=self.hidden_layer(x)
        x=self.hidden_relu(x)
        x=self.output_layer(x)
        sigmoid_x=self.sigmoid(x)
        return(sigmoid_x)
scaler=StandardScaler()#표준화 
#표준화 작업(fit_transform)
X_scaler=scaler.fit_transform(X_train)#X_train
X_scaler_t=scaler.fit_transform(X_test)#X_test(어쩃든 정보가 있는 자료니깐 표준화 스케일링)
#mps(필수 사항은 아님)
if torch.backends.mps.is_available():
    mpsT=torch.device('mps')
    print('GPU accessed')
else:
    print('CPU accessed')
#tensor에 담아주기
X_ten=torch.tensor(X_scaler,dtype=torch.float).to(mpsT)
X_ten_t=torch.tensor(X_scaler_t,dtype=torch.float).to(mpsT)
#Y도 똑걑이 담아주기
Y_ten=torch.tensor(Y_train,dtype=torch.float)
Y_ten_t=torch.tensor(Y_test,dtype=torch.float)
#doctor
doctor_dl=EMRmodule().to(mpsT)
#채점 및 반복학습(epochs) 겸 테스트
criterion=nn.CrossEntropyLoss()
#교차엔트로피손실:이진교차엔트로피랑은 다르게 답을 3가지 이상 출력을 할 때 필요한 채점 기술
optimizer=optim.Adam(doctor_dl.parameters(),lr=0.01)
#Adam이라는 최점단 기술을 이용해 학습률(보폭)을 0.01로 설정하고, 딥러닝 의사 매개변수를 0에 수렴할 수 있도록 최적의 설정 진행
for epochs in range(1000):
    optimizer.zero_grad()#누적 데이터 삭제->누적 데이터가 있을 시 기존에 있던 값과 새로 학습한 값을 계산해 방향이 엉뚱하게 흘러감
    prediction=doctor_dl(X_ten)
    loss=criterion(prediction,Y_ten)
    loss.backward()
    optimizer.step()
#DL result(update된 의사를 사용해보자!)
with torch.no_grad():#계산 금지,오답만 빠르게 체크하기
    dl_output=doctor_dl(X_ten_t)
    pred_dl=torch.argmax(dl_output,dim=1)
    print(f'Deep Learning Accuracy: {accuracy_score(Y_ten_t.cpu(),pred_dl.cpu())*100:.1f}%')
    #정확도 테스트를 위해 다시 gpu->cpu로 정보 옮기기
#argmax:가장 큰 값이 존재하는 index를 가져오기
#dim=AI가 데이터를 읽는 방향 설정->dim=1은 1행을 기준으로 argmax를 적용해 가장 큰 idx를 불러옴
#final result
target_idx=0
real_answer=Y_ten_t[target_idx].item()#2array
#ml pred_proba
single_x_ml=X_test.iloc[[target_idx]]#target_idx번째 정보 가져오기
prob_ml=tuning_rf.predict_proba(single_x_ml)[0]#가장 높게 나온 값에서 환자들의 0,1,2그룹에 몇 퍼센트를 가지고 속해있나
#dl 
single_x_dl=X_scaler_t[[target_idx]]
single_x_dl_tensor=torch.tensor(single_x_dl,dtype=torch.float).to(mpsT)
#다시 gpu환경에서 돌리기
#with은 개인공간에서 진행했던 '정확도' 테스트였다->이번 과정은 target_idx정보만 출력하기 위한 것
doctor_x_dl=doctor_dl(X_ten_t)
prob_dl=torch.nn.functional.softmax(doctor_x_dl,dim=1)[0].detach().cpu()
#softmax는 어떠한 형태(양/음수)로 나오든 상관없이 0~1(100%)로 치환해서 범위 내에 출력된다
#detach:성능 측면에서 부담을 덜어주는 역할
print(f'Real answer: {real_answer}idx')
print(f'ML Real answer(0): {prob_ml[0]*100:.1f}%')
print(f'DL Real answer(0): {prob_dl[0]*100:.1f}%')
#순수 숫자로 이루어진 배열에서 [0]을 통해 정보를 가져오기 (0,1,2) 중 0
#예제1~7 26.5.11~5.25