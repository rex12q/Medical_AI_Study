import torch #pytorch 라이브러리 불러오기
import torch.nn as nn #neural network
import matplotlib.pyplot as plt
patients_data=torch.linspace(-5,5,100) #선형 공간 생성(-5~5까지 100개로 쪼갬)
try:
    if torch.backends.mps.is_available(): #Macbook전용 mps 사용 
        mps_device=torch.device('mps') #mps
        fast_machine=patients_data.to(mps_device)#환자들 정보 to를 이용해 mps로 
        print('test accessed! (Result: GPU)')
    else:
        print('test accessed! (Result: CPU)')
except Exception as e:
    print(f'ERROR: {e}') #터졌을 경우
#주의:CPU->GPU는 해당 if절에서만 변경이 되었을 뿐,실제로는 GPU가 아닌 CPU로 계산이 되기에 별도로 설정을 해야함
#CPU->GPU
patients_gpu=patients_data.to(mps_device)#연산할 공간 옮기기
#ReLU,Sigmoid
relu_filter=nn.ReLU().to(mps_device) #정상값(0이하의 값들은 전부 drop)
relu_filter_result=relu_filter(patients_gpu) 
clean_output=relu_filter_result.cpu()#연산은 gpu에서 했으나,그래프에 나타낼 때는 무조건 cpu로 연산을 해야함 
sigmoid_filter=nn.Sigmoid().to(mps_device)#컴퓨터가 이해할 수 있도록 확률 변환(0~1)
sigmoid_filter_result=sigmoid_filter(patients_gpu)
clean_probability=sigmoid_filter_result.cpu()
plt.figure(figsize=(14,6))
plt.subplot(1,2,1) #(행:1,열:2,index:1)
plt.plot(patients_data.numpy(),clean_output.numpy(),color='red') #(X:환자 정보,Y:필터)
plt.title('ReLU filter')
plt.subplot(1,2,2) 
plt.plot(patients_data.numpy(),clean_probability.numpy())#(X:환자 정보,Y:필터)
plt.title('Sigmoid filter')
plt.show()