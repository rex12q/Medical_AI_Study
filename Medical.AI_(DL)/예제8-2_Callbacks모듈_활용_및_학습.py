X=1
y=2
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
#DL/예제 8 ModelCheckpoint에 대한 이해
from tensorflow.keras.callbacks import ModelCheckpoint #모듈은 keras 내장 도구,callbacks를 사용
#ModelCheckpoint: 모델이 학습을 하는 도중 가장 성능이 높게 나왔던 학습 순간을 저장(Checkpoint)
mcheck=ModelCheckpoint(monitor='val_loss',filepath='best_model.keras',save_best_only=True,verbose=1)
#mointor: 지표이므로 어떠한 방식으로 나타낼 수 있음
#filepath: Best Moment를 저장할 모델 파일 경로 (보통 모델 확장자명과 함께 가명으로 나타냄)
#save_best_only: 가장 우수한 모델 성능이 나왔을 때 저장할 수 있는 기능 
#verbose: 터미널에 나타낼 수 있게 해주는 성능 기록지 (앞에 조건이 더해지면 매 순간마다 기록을 표시하진 않음)
#verbose는 특이하게 True,False를 안 씀 -> 0:침묵모드,1:진행 표시줄,2:epoch당 깔끔하게 한 줄씩만 출력

#모델에 직접 활용
from tensorflow.keras.models import Sequential #신경망
from tensorflow.keras.layers import Dense,Dropout #있는 노드를 전부 깔끔하게 보내기,과적합을 방지하기 위한 삭제 시스템
model=Sequential([
    Dense(64,activation='relu'),
    Dropout(0.3) #무작위 30프로만 드랍(젼원 끄기)해서 강제로 학습시키기
    #... 신경망이 전부 구성되었다고 가정 
])
#모델 학습 (텐서플로우 기준) 
history=model.fit(X_train,y_train,epochs=50,batch_size=10,validation_data=(X_test,y_test),callbacks=[mcheck])
#callbacks에 ModelCheckpoint가 들어감 
#학습 데이터, 이포크(반복 학습 횟수):50, 배치 사이즈(~만큼 풀고 채점(후 반복)):10,검증 데이터: 시험 데이터
#모델은 50번동안의 검증, 검증할 동안 옮길 데이터 수 10, 검증할 때 참고 검증 데이터는 시험 데이터, 콜백을 이용하여 모델의 최고 순간 기록 및 저장 가능
########################################################################################################################################################

#DL/예제 8 EarlyStopping에 대한 이해
#파이토치는 직접 사용자가 수동으로 전부 설계가 가능함 -> class,def,(if/else)를 이용하여 모듈을 상세하게 조정
from tensorflow.keras.callbacks import EarlyStopping #모듈은 keras 내장 도구,callbacks를 꺼내서 씀
import torch
class EarlyStopping: #이름: early_stopping
    #초기 단계: 셀프 인식, 인내심=14, 설명=x, 경로(저장 이름)='best_model.pth' (pth: 업데이트 후 토치 확장자명)
    #초기 단계는 말 그대로 백지 상태여야 함. 그렇기에 값이 지정된 건 아무 것도 없음(patience 제외)
    def __init__(self,patience=14,verbose=False,path='best_model.pth'): 
        self.patience=patience #인내심 = 인내심
        self.verbose=verbose #터미널 설명 출력
        self.counter=0 #경고 횟수 누적을 위해 0부터 시작
        self.best_score=None #폼 미쳤을 때 = 
        self.early_stop=False #얼리스탑 x
        self.path=path #경로 = 경로 (이름)
    #call을 이용하여 섹션 나누기
    def __call__(self,val_loss,model): 
        #첫번째 epoch
        if self.best_score is None: #첫 시험임을 알림 (None)
            self.best_score = val_loss # 최고 점수 = 검증 손실
            #save_checkpoint 함수에 검증 손실, 모델 기입
            self.save_checkpoint(val_loss,model) #이로써 비교해야 할 초기 점수(val_loss)와 모델한테 자료 전달(=저장)
        #신기록 달성하지 못했을 때
        elif val_loss >= self.patience: #횟수가 14보다 높거나 같을 때
            if self.verbose:
                print('')
            if val_loss >= self.patience: 
                self.early_stop=True # 
        #신기록 달성했을 때
        else:
            self.best_score=val_loss #초기 점수에서 가장 높은 순간이 나왔을 때 저장
            self.save_checkpoint(val_loss,model) #위와 마찬가지로 순간을 저장
            self.counter=0 #누적할 필요 없음
    def save_checkpoint(self,val_loss,model):
        if self.verbose:
            print('')
        #state_dict: 해당되는 모델의 가중치(기여) 값, 경로(이름),출력
        torch.save(model.state_dict(),self.path) 
#나누어서 EarlyStopping 활용
early_stopping=EarlyStopping(patience=14,verbose=True)