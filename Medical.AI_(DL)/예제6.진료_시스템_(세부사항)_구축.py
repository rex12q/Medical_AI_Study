#나이,BMI,SYSBP,DIABP..etc이렇게 단위가 다른 값들을 동일선상에 두고 값을 출력할 시 심각한 논리적 오류가 발생한다
#그러므로 표준화 작업을 위해 StandardScaler 활용
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
class DLmodule(nn.Module):
    def __init__(self):
        super().__init__()#부모 클래스 잊지 말기
        #input
        self.input_layer=nn.Linear(in_features=4,out_features=16)#입력4,출력16
        self.input_relu=nn.ReLU()#0이하 값들 drop
        #hidden
        self.hidden_layer=nn.Linear(in_features=16,out_features=8)#16받고,출력8
        self.hidden_relu=nn.ReLU()
        #output
        self.output_layer=nn.Linear(in_features=8,out_features=1)#8받고,최종출력1
        self.sigmoid=nn.Sigmoid()
    def forward(self,user_data):#길안내,임의로 데이터 경로를 나타내기
        x=self.input_layer(user_data)
        x=self.input_relu(x)#정보를 x에 넣고 relu로 0이하 값 drop
        x=self.hidden_layer(x)
        x=self.hidden_relu(x)
        x=self.output_layer(x)
        # x=self.output_relu(x)모든 값이 '-'이면 relu로 인해 값 출력이 안될 수도 있음
        sigmoid_x=self.sigmoid(x)
        return sigmoid_x
#순서: [나이, BMI, 혈압, 혈당]
raw_X = [
    [21, 23.0, 80,  85],  # 아주 건강한 사람
    [45, 27.5, 135, 110], # 당뇨 위험군 사람
    [60, 31.0, 150, 140], # 중증 당뇨 사람
    [25, 21.0, 110, 90],  # 건강한 사람
    [55, 29.0, 145, 125]  # 당뇨 사람
]
#정답지
raw_Y=[[0],[1],[1],[0],[1]]
#표준화 스케일링(체급 맞추기)
standard=StandardScaler()
standard_x=standard.fit_transform(raw_X)
#pytorch.tensor
x_info=torch.tensor(standard_x,dtype=torch.float)#DL은 무조건 float형태를 받아야함
y_info=torch.tensor(raw_Y,dtype=torch.float)
#doctor!
doctor=DLmodule()
#BCE(Binary Cross Entropy,이진 교차 엔트로피):정답을 맞췄을 경우 0,틀렸을 경우 1에 가까워짐
#MSE를 쓰게될 경우 경사값이 완만해져서 backward()(역추적)을 할 때 0으로 수렴하는 길을 잃어버리고 모델이 퍼짐
bce=nn.BCELoss()
#adam(Adaptive Momentum,적응형 모멘텀):GD와 보폭의 한계를 극복한 최첨단 기술(상세 설명은 아래)
optimizer=optim.Adam(doctor.parameters(),lr=0.01)
#채점,역추적,성능테스트
print('셋업 중..')
for epochs in range(1000):
    prediction=doctor(x_info)
    loss=bce(prediction,y_info)
    #zero_grad(전이랑 다른 위치 조정):예제2에서는 앞전에 썼지만, 이번 예제는 문제를 먼저 풀고 채점을 한 다음에 초기화
    optimizer.zero_grad
    loss.backward()#역추적(편미분)
    optimizer.step()#역추적 성공 후 업데이트
#result
print('셋업 끝')
print('1번~5번 사람까지 모든 결과를 공개합니다')
result=doctor(x_info)
print(f'''아주 건강한 사람: {result[0].item()*100:.1f}%
        당뇨 위험군 사람: {result[1].item()*100:.1f}%
        중증 당뇨 사람: {result[2].item()*100:.1f}%
        건강한 사람: {result[3].item()*100:.1f}%
        당뇨 사람: {result[4].item()*100:.1f}%''')

#adam:경사하강법과 보폭 수정의 한계를 극복한 기술,
#방향(Momentum):0으로 수렴하는 길로 가되, 가는 방향으로 관성을 줌
#보폭:많이 바뀐 가중치 값은 미세 조정,적게 바뀐 가중치 값은 크게 조정
#상황에 맞게 0에 수렴할 수 있도록 최대한의 효율을 자랑
#설계
#데이터 준비->신경망 설계->전처리(표준화)->학습->최적화(adam)함수 셋팅->훈련 루프->결과 도출
