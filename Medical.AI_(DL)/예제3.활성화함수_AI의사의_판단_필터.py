import torch
import torch.nn as nn 
import matplotlib.pyplot as plt

#가상의 환자 점수
patients_score=torch.linspace(-5,5,100)
#lin(ear)space선형공간:(start,end,steps)-> -5~5까지 100개로 쪼개라 -> 100개로 쪼개진 1차원 배열 생성(딥러닝만의 복잡한 기능이 내재되었기에 숫자 배열은 x)
#곡선 그래프를 그릴 때 꼭 필요한 도구

#ReLU(Rectified Linear unit,정류 선형 단위):활성화 함수,정상치 필터,0이하의 값들은 drop
relu_filter=nn.ReLU()
clean_output=relu_filter(patients_score)#필터 씌워주기

#Sigmoid(ML예제3 참고):0부터 1사이로 확률을 나타내 줄 수 있는 변환기
sigmoid_filter=nn.Sigmoid()
clean_result=sigmoid_filter(patients_score)

#result
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
#subplot(행의 개수,열의 개수,index시작 위치):한 화면에 여러 개의 그래프를 배열 형태로 나누어 그리는 기능
plt.plot(patients_score.numpy(),clean_output.numpy(),color='blue')
plt.title('ReLU: Minus=0 (Tumor size)') #ReLU필터를 통해 종양 크기를 출력

plt.subplot(1,2,2) #index(2)
plt.plot(patients_score.numpy(),clean_result.numpy(),color='red')
#딥러닝에 있는 미분,신경망..등 기능을 전부 제외하고 순수 숫자로 구성된 배열로만 나타내기
plt.title('Sigmoid filter result|Range(0~1)|Diease Risk')
plt.show()