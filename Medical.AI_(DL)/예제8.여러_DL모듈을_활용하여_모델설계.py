import torch
import torch.nn as nn #neural network
import torch.optim as optim #최적화 기능 모듈 제공
import numpy as np
from tensorflow.keras.models import Sequential #Layer모델 함수
#Dense: 모든 노드와 연결되어 있는 연결되어 있는 상태
#Dropout: 과적합 방지, 특정 노드에 의존할 수 없도록 모든 노드 학습
# from tensorflow.keras.layers import Dense,Dropout 
#EarlyStopping: 과적합이 심해질 경우 제지해주는 역할 (Validation Loss이 늘어나면 제지)
#ModelCheckpoint: 모델 학습 중 가장 성능이 좋았던 순간의 모델을 파일로 자동 저장
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#순서: [나이, BMI, 혈압, 혈당]
raw_X = [
    [21, 23.0, 80,  85],  # 아주 건강한 사람
    [45, 27.5, 135, 110], # 당뇨 위험군 사람
    [60, 31.0, 150, 140], # 중증 당뇨 사람
    [25, 21.0, 110, 90],  # 건강한 사람
    [55, 29.0, 145, 125]  # 당뇨 사람
]
#정답지
raw_y=[[0],[1],[1],[0],[1]] #이진 분류
#다층 퍼셉트론  (Multi layer perceptron)
class MLPset(nn.Module):
    def __init__(self): #init 초기 단계 설정
        super().__init__() #상위 부모 클래스 가져오기
        self.input_layer=nn.Linear(4,16) #col 4개에 맞춰서 in_features설정
        self.input_relu=nn.ReLU() #0이하 값 drop
        self.hidden_layer=nn.Sequential(
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32,8),
            nn.ReLU(),
            nn.Dropout(0.2)
        ) #Sequential 신경망 구성 
        self.output_layer=nn.Linear(8,1)
        self.sigmoid_x=nn.Sigmoid() #이진 분류를 위해 시그모이드 활용
    def forward(self,data_load): #데이터 흐름과 방향(네비게이션 역할)
        x=self.input_layer(data_load)
        x=self.input_relu(x)
        x=self.hidden_layer(x)
        x=self.output_layer(x)
        x=self.sigmoid_x(x)
        return x
#정보,테스트 분할
X_train,X_test,y_train,y_test=train_test_split(raw_X,raw_y,test_size=0.2,random_state=7)
st=StandardScaler()
X_train=st.fit_transform(X_train) #학습->변환
X_test=st.transform(X_test) #학습->변환
#정보 텐서에 담아주기
X_info=torch.tensor(X_train,dtype=torch.float) #DL은 float만 취급
X_t_info=torch.tensor(X_test,dtype=torch.float)
#-1:데이터 개수에 맞춰서 알아서 전부 가져오기, col=1 -> [[]]
y_info=torch.tensor(y_train,dtype=torch.float).view(-1,1) #(row,col)
y_t_info=torch.tensor(y_test,dtype=torch.float).view(-1,1)
#의사양반
doctor=MLPset()
#Binary Cross Entropy
bce=nn.BCELoss() #True면 0 False면 1(MSE는 경사값이 완만해져서)
#stop과 checkpoint
# #지표: 검증 손실, 인내심:14 -> 가장 좋은 검증 손실이 나왔을 때, 최고 기록(가장 오답이 적을 때)을 넘지 못 할 경우
# estop=EarlyStopping(monitor='val_loss',patience=14) 
#가장 폼이 좋았던 순간을 CHECK!!! (업데이트 후 h5 -> keras) 폼 미쳤던 순간만 저장, verbose=저장할 떄 마다 터미널 알람 설정
# mcheck=ModelCheckpoint(monitor='val_loss',filepath='best_model.keras',save_best_only=True,verbose=1)
#optimizer Adam 세부 설정
optimizer=optim.Adam(doctor.parameters(),lr=0.01) 

#class: EarlyStopping (얼리스탑 클래스를 따로 분류하여 깔끔하게 설계)
class EarlyStopping:
    #인내심:14,verbose=저장할 떄 마다 터미널 알람 설정,pth(pytorch 확장자)
    def __init__(self,patience=14,verbose=False,path='best_doctor_model.pth'):
        self.patience=patience #인내심 테스트
        self.verbose=verbose #최고 기록 갱신 시 터미널 알람 
        self.counter=0 #참은 횟수 카운터
        self.best_score=None #최고 점수 (가장 낮은 loss)
        self.early_stop=False #조기 종료 스위치
        self.path=path #모델 저장 경로(ModelCheckPoint)

    def __call__(self, val_loss, model): #__call__을 이용하여 섹션 나누기
        # 첫 번째 에포크일 때: 비교할 점수가 없기에 현재 점수를 최고 점수로 기록
        if self.best_score is None: 
            self.best_score = val_loss #최고 점수=검증 손실
            self.save_checkpoint(val_loss, model) 
        #신기록 달성 못했을 때: 성적이 그대로거나 그 이하
        elif val_loss >= self.best_score:
            self.counter += 1 #1회씩 누적을 해야 patience 설정값에 만족할 수 있음->강제 종료 (실망했으니 1점씩 추가)
            if self.verbose:
                print(f'[EarlyStopping] 참을성: {self.counter} / {self.patience}')
            if self.counter >= self.patience: #카운터가 인내심 횟수보다 많거나 같을 경우 얼리스탑 적용
                self.early_stop=True #
        #신기록 달성 시
        else:
            self.best_score=val_loss
            self.save_checkpoint(val_loss,model) #가장 폼이 좋았던 val_loss,model을 저장하기
            self.counter = 0 #신기록 갱신, 그래서 0으로 다시 리셋
    #save_checkpoint 함수
    def save_checkpoint(self,val_loss,model):
        if self.verbose:
            print('[모델 저장됨!] 최고 기록을 저장합니다')
        #state_dict:가중치 값 출력, path 저장
        torch.save(model.state_dict(),self.path) #가장 성능이 좋았던 순간을 저장하여 저장한 내용을 씀
#EarlyStopping 모듈처럼 쓰기
early_stopping=EarlyStopping(patience=14,verbose=True)
#pytorch에는 compile기능이 없기에 모듈을 밖에다가 두고 for문으로 반복 훈련
epochs=1000
best_val=np.inf # 모델의 최고 성능 순간을 기록하기 위해 inf(inity)로 설정
for epoch in range(epochs): #for 이름 epoch로 설정
    studying=doctor(X_info) #훈련용 데이터로 비교
    loss=bce(studying,y_info)
    optimizer.zero_grad() 
    loss.backward() #역추적(편미분,어디서 오답이 발생했나)
    optimizer.step() #최신화

    #모델(의사) 평가 테스트 (실전용 모드)
    doctor.eval() #.eval() (실전 모드)
    with torch.no_grad(): #메모리 낭비 방지(복잡한 그래프 그리기 방지,시험에만 몰두)
        pred=doctor(X_t_info) #실제 테스트
        loss_t=bce(pred,y_t_info) 
    early_stopping(loss_t.item(),doctor) #bce기반 시험을 치른 doctor가 early_stopping에게 제출

# 진행 상황을 터미널에 출력
    if (epoch + 1) % 10 == 0: #0이아닌 1부터 시작, 10으로 나눠서 나머지가 0일 경우 출력 (출력 시 10 간격으로 출력이 됨)
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {loss.item():.4f} | Val Loss: {loss_t.item():.4f}")

    #조기 종료 및 모델 저장 
    #14번을 넘었기에 더 이상의 과적합을 봐주지 않고 바로 조기 퇴근행으로 보냄
    if early_stopping.early_stop:
        print('Early stop 발생!')
        break
#코드를 작성하고 실행할 시 출력되는 것은 Train Loss: 0.0000 | Val Loss: 0.0000 과 epoch가 10단위로 떨어져서 출력이 된다
#신경망 구조에 비해 모델이 학습할 학습량이 압도적으로 적기 때문에 극단적인 Overfitting이 일어나는 것을 확인할 수 있다
#mcheck,earlystopping은 이미 class와 def를 이용하여 만듦 (세부 조정)