#본 EMR데이터는 XX대학교  극히 일부분 '틀'(내용X)을 가지고 진행했음을 알립니다
# 절대 실제 환자의 기록을 토대로 학습을 하지 않았으며 이를 가지고 유포하거나 타인에게 공유할 시, 국내 의료법에 위반된다는 걸 인지하고 있습니다.
####################################################################################################################################################################################
#예제 7의 주제는 동일하지만, 한 번에 많은 양의 데이터를 cpu->gpu로 옮기려고 할 때 메모리 할당량의 한계로 batch_size를 이용해 데이터를 옮길 것
#TensorDataset,DataLoader를 이용|예제 7 복습 
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#TensorDataset:파이토치에서 독립변수,종속변수 텐서를 하나로 묶어주는 역할 클래스
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
csv_load=pd.read_csv('csv(DL)/ML,DL,test_file.csv')
#nn
class EMRtest(nn.Module):
    def __init__(self):
        super().__init__()
        #input
        self.input_layer=nn.Linear(6,24)
        self.input_relu=nn.ReLU()
        #hidden
        self.hidden_layer=nn.Linear(24,12)
        self.hidden_relu=nn.ReLU()
        #output
        self.output_layer=nn.Linear(12,3)#0,1,2
    def forward(self,ex_data):
        #input
        x=self.input_layer(ex_data)
        x=self.input_relu(x)
        #hidden
        x=self.hidden_layer(x)
        x=self.hidden_relu(x)
        #output
        x=self.output_layer(x)
        return x
#gpu
mpsT=torch.device('cpu')
print('CPU accessed')#가벼운 신경망은 cpu..
X=csv_load[['Gender','Age','WC','BMI','SBP','GLU']]
#Y
risk_count=(
    (csv_load['BMI']>=30).astype(int)+
    (csv_load['SBP']>=130).astype(int)+
    (csv_load['GLU']>=126).astype(int)
)
csv_load['p_result']=np.where(risk_count==0,0,np.where(risk_count==1,1,2))#최종 ouput:2
Y=csv_load['p_result']
#train
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
#compose column
gen=['Gender']
other=['Age','WC','BMI','SBP','GLU']
gen_col=Pipeline([
    ('gen_simp',SimpleImputer(strategy='most_frequent'))
])
other_col=Pipeline([
    ('other_simp',SimpleImputer(strategy='median')),
    ('stad',StandardScaler())
])
total_col=ColumnTransformer(
    transformers=[('gen_total',gen_col,gen),
    ('other_total',other_col,other)],
    sparse_threshold=0#희소행렬 완전 차단!
)
#standardscaler(중요)
stad=StandardScaler()
X_scaler=total_col.fit_transform(X_train)#학습+스케일링
X_scaler_t=total_col.transform(X_test)#학습 제외,스케일링만 fit_transform(DataLeak!)
#tensordataset(tensor로 담아주기)
train_set=TensorDataset( #학습용 데이터X와 학습용 정답지Y로 묶어주기
    torch.tensor(X_scaler,dtype=torch.float).to(mpsT), #실수 형태로 바꿔주기
    #values로 리스트를 벗기고,순수 숫자만 남겨둠|long=int64
    torch.tensor(Y_train.values,dtype=torch.long).to(mpsT)
    )
train_load=DataLoader(train_set,batch_size=10,shuffle=True)
#train_set에서 배치 사이즈는 10으로 그리고 무작위 정보 섞어서 gpu로 보내기
#fit,loss
doctor_dl=EMRtest().to(mpsT)
cel=nn.CrossEntropyLoss()
optimizer=optim.Adam(doctor_dl.parameters(),lr=0.01)
for epochs in range(1000):
    for batch_x,batch_y in train_load:#배치 설정이 끝난 후 gpu로 보내기
        batch_x=batch_x.to(mpsT)
        batch_y=batch_y.to(mpsT)
        optimizer.zero_grad()
        prediction=doctor_dl(batch_x)
        loss=cel(prediction,batch_y)
        loss.backward()
        optimizer.step()
#with(DL accuracy)
with torch.no_grad():#불필요한 계산 기능 배제
    X_scaler_t_ten=torch.tensor(X_scaler_t,dtype=torch.float).to(mpsT)#텐서로 담아주기
    dl_output=doctor_dl(X_scaler_t_ten)
    dl_pred=torch.argmax(dl_output,dim=1)
    #argmax를 통해 가장 큰 값을 가진 인덱스를 가져와 dim으로 첫번쨰 환자로 시선을 바꿔 인데스를 가져오기
    print(f'DL model accuracy: {accuracy_score(dl_pred.cpu().numpy(),Y_test)*100:.1f}%')
    #순수 숫자 형태,cpu로 옮겨서 결과 출력
#result
target_idx=0
#일치하는 인덱스에 있는 정보 가져오기(iloc)->[['','',...]](줄번호로 가져오기)
real_answer=Y_test.iloc[target_idx]
X_last_t=X_scaler_t[target_idx].reshape(1,-1)#(행:1,열:데이터에 맞게 알아서 가져옴)
X_last_t_ten=torch.tensor(X_last_t,dtype=torch.float).to(mpsT)#텐서에 담아주고 GPU행
#doctor with tensor|위에서는 정확도 확인용,이 파트에선 예측 가능성!
doctor_dl_ten=doctor_dl(X_last_t_ten)
#함수 softmax로 0~1(0~100%)로 컴퓨터가 이해할 수 있도록 바꾸기
prob_dl=nn.functional.softmax(doctor_dl_ten,dim=1)[0].detach().cpu().numpy()#dim=1로 시선 옮기기([0]이 있기에 첫번째 칸만 집중공략)
#detach로 계산 흔적 전부 지우고,값 출력이 가능하도록 GPU->CPU로 변환,numpy로 순수 숫자 형태로 남겨두기,[0]은 행렬 첫번째 환자칸(타겟),마지막으로 dim이 전부 다 합쳤을 때 100이 나오도록 계산
#[ 5.0,  1.2, -3.1]-Dim->[ 80.0%,  19.9%,   0.1% ]
print('DL Result')
print(f'Real answer: {real_answer}idx')
print(f'Normal(0): {prob_dl[0]*100:.1f}%')
print(f'Danger(1): {prob_dl[1]*100:.1f}%')
print(f'High-risk(2): {prob_dl[2]*100:.1f}%')


#ERROR설명
#X_last_t=X_scaler_t[target_idx].reshape(1,-1)
#"__getitem__" 메서드가 "spmatrix" 형식에 정의되지 않았습니다. 
#이 에러는 컬럼을 표준화 스케일링,결측치 처리를 한 과정이 있기에, 컴퓨터는 용량 이슈로 인해 압축을 진행-spmatrix>
#spmatrix(sparse matrix:희소행렬):희소행렬은 행렬을 차지하고 있는 대부분의 값이 0일 때를 의미함.->표준화 스케일링 과정에서 1보다 0의 값이 더 많기에 그냥 컴퓨터가 희소행렬로 판결(압축파일과정)
#평범한 2차원 배열이 아닌 압축파일이 희소행렬이라서 __getitem__(대괄호) 오류 발생! -해결책-> toarray():압축(괄호)풀기(벗기기)->이러면 원래 형태의 2차원 배열(reshape(1,-1))형태로 돌아오게됨
#또는 멋대로인 '열변환기' 괄호 안에 sparse_threshold=0으로 해결이 가능(희소x,무조건 일반 배열로!!)
