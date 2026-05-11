import torch #텐서 엔진
import torch.nn as nn #Neural Network(nn): 신경망의 뼈대,자세한 내용은 아래 참고
import torch.optim as optim #optim: 최적화 성능 제공
#데이터(->텐서)
X=torch.tensor([[1.0],[2.0],[4.0]]) #공부 시간 
Y=torch.tensor([[[50.0],[68.0],[89.0]]]) # 성적
#인공신경망 생성 (입력:1개, 출력:1게),(결괴를 늘린다고 무조건 모델의 성능이 올라가는 것은 아니다. 연관이 있어야 UP!!)
brain_model=nn.Linear(in_features=1,out_features=1)#단순한 데이터, linear(선형)을 이용해 y=Wx+b (W=Weight) 1차 방정식
#정답 체크 (안전 장치)
criterion=nn.MSELoss()#MSE: 오차에 제곱을 해서 평균을 내는 방식(제곱을 하기에 오차가 클수록 모델의 전체적인 성능 저하)
optimizer=optim.SGD(brain_model.parameters(), lr=0.01) #SGD 아래서 참고,lr(learning rate):학습률,보폭을 설정해 0에 수렴할 수 있도록 도와줌
#parameters(매개변수)는 모델 내부의 가중치(W)와 편향(b)를 의미한다.->출력값이 완전 다르게 나왔을 경우 W와 b를 건드려 조정
#Training loop(예측->오차출력->역전파(편미분 과정)->업데이트)
for d in range(100):
    optimizer.zero_grad #zero_grad:pytorch는 이전에 학습했던 정보가 있기에 진짜 성능을 테스트 하기 전 한 번 지워주는 기능(리셋)
    prediction=brain_model(X) #성능 테스트
    loss=criterion(prediction,Y) #정답 맞춰보기(ML 다뤘을 때 생각하기..)
    loss.backward() #backward(역전파):방금 푼 loss에서 오차(오답)가 어디서 발생했는지 수학적(편미분)으로 접근,(오차(출력)-(편미분)>은닉-(편미분)>입력)역방향
    optimizer.step() #step:편미분값에 학습률(lr)을 곱하여 가중치 값을 수정->0에 수렴할 수 있도록 미세조정
#결과
print(f'1시간 공부했을 때 AI의 점수 예측: {brain_model(torch.tensor([[1.0]])).item():.2f}') #tensor로 감싸주기
print(f'2시간 공부했을 때 AI의 점수 예측: {brain_model(torch.tensor([[2.0]])).item():.2f}')
print(f'4시간 공부했을 때 AI의 점수 예측: {brain_model(torch.tensor([[4.0]])).item():.2f}')

#linear Regression(선형 회귀): 데이터 흐름의 직선(y=ax+b)을 찾고, 과거의 데이터를 바탕으로 미래의 수치값을 예측할 때 사용
#Neural Network(nn): 신경망의 뼈대 (층layer, 손실함수 loss function, 활성화 함수 activation function)..등 모든 모듈이 nn안에 정의됨
#시그모이드 함수=활성화 함수(Medical ML 예제3 참고)
#인공신경망 생성 파트
#딥러닝 구조: 입력층(input layer)->은닉충(hidden layer)->출력층(output layer)
#후에 다룰 딥러닝을 튜닝할 시 linear블록을 수백 개 겹겹이 쌓아올려 Sigmoid(예제3)같은 활성화 함수를 끼워 넣어 '비선형'적인 모델로 바꾸어 똑똑하게 만듦
#SGD(Stochastic Gradient Descent(확률적 경사 하강법)):오차0에 수렴할 수 있도록 하는 명령어,미분값을 찾아 그 방향으로 감
#Training loop
#backward:모델이 문제를 풀고 난 뒤에 생긴 오차를 편미분으로 접근-편미분(o),미분(x)
#편미분:여러 개의 가중치가 존재할 때 어떤 가중치가 잘못됐는 지를 모르니,모든 **각 가중치의 값을 동시에 수정**하며 오차를 조정->편미분값에 학습률을 곱하면 미세조정이 가능
#오차 0에 수렴하기 위해 w1값 수정(아직 완벽하게 0에 수렴x),동시에 w2을 수정해 오차 0에 수렴...(최대한 0에 수렴할 수 있는 값들을 구해 출력)
