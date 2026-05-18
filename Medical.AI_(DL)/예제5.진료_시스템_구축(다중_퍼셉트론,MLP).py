#ML을 다뤘던 공간에 DL모델이 추가됨
#Layer1)in_features=4,out_features=16|(Layer2)in_features=16,out_features=8(16개에서 8개로 정보 압축)
#Layer3)in_features=8,out_features=1(8개->최종 출력층에선 1개로 결과 출력) (구조)
import torch
import torch.nn as nn
#설계|#처음은 정보 4개가 들어오지만,1층에서 16개로 쪼개져서 2층에서 8개로 압축을 하고 최종적으론 3층에서 1개로 압축을 해서 정보를 출력
class PredictorAI(nn.Module):#nn.Module:사용자 만의 인공지능 신경망을 설계하기 위한 레고판
    def __init__(self):
        super().__init__()#super(){부모}:상위 모듈을 부르는 명령어
        #Layer1
        self.doctor_basic=nn.Linear(in_features=4,out_features=16)
        self.relu_basic=nn.ReLU()#0이하 값 drop
        #Layer2
        self.doctor_med=nn.Linear(in_features=16,out_features=8)
        self.relu_med=nn.ReLU()
        #Layer3
        self.doctor_hard=nn.Linear(in_features=8,out_features=1)#결과 도출은 오직 하나로
        self.relu_hard=nn.ReLU()
        #sigmoid(0~1)
        self.sigmoid=nn.Sigmoid()
    #길안내
    def forward(self,patients_data):#forward:새로 기입된 데이터가 앞으로 흘러가는 길(그걸 안내해줌)
        #input
        x=self.doctor_basic(patients_data)#데이터 기입 시,첫번째 순서
        x=self.relu_basic(x)#0이하 drop
        #hidden
        x=self.doctor_med(x)#받고
        x=self.relu_med(x)
        #output
        x=self.doctor_hard(x)
        # x=self.relu_hard(x) 
        #final probability
        x_sigmoid=self.sigmoid(x)
        return x_sigmoid #최종 확률값 리턴
#사용자 정보
def UserInfo(prompt,minval,maxval,is_float=True):
    while True:
        try:
            user_v=float(input(prompt)) if is_float else int(input(prompt))
            if minval <= user_v <= maxval:
                return(user_v)
            print('범위를 넘어버린.')
        except Exception as e:
            print(f'오류: {e}')
age = UserInfo('나이를 입력하세요 (0~120): ', 0,120,is_float=False)
bmi = UserInfo('BMI 수치를 입력하세요 (10~50): ',10,50)
bp = UserInfo('혈압을 입력하세요 (80~200): ',80,200)
glu = UserInfo('혈당 수치를 입력하세요 (50~300): ',50,300)
user_list=[age,bmi,bp,glu]#tensor로 담기 위해 리스트로 묶기
#tensor
user_tensor=torch.tensor([user_list],dtype=torch.float)#DL은 float만 인식하기에 모든 데이터 유형을 float으로 바꿈
#CPU,GPU test
if torch.backends.mps.is_available():
    mps_model=torch.device('mps')
    fast_model=user_tensor.to(mps_model)
    print('GPU test accessed!')
else:
    print('CPU test accessed!')
#doctor!
doctor=PredictorAI().to(mps_model)#똑같은 GPU공간에서 실행
user_tensor_gpu=user_tensor.to(mps_model)#GPU로 돌리기
print(f'사용자 텐서가 {mps_model} 무사히 정보를 가지고 옮김.')
print(f'최종 텐서 모양: {user_tensor_gpu.shape}')
result=doctor(user_tensor_gpu)
print(f'사용자의 당뇨 발병 확률: {result.item()*100:.2f}%')
