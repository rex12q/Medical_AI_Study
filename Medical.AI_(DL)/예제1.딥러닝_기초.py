import torch
#torch:딥러닝의 메인 엔진 파이토치(PyTorch) 라이브러리를 불러옴
#tensor:일차원부터 다차원 배열(numpy,csv와 동일)을 통칭함,'Rank'로 분류가 되며 스칼라,벡터,행렬,Rank3..로 분류,다만 GPU를 쓰는 극한 작업이라는 차이
data=[[1,2],[3,4]] 
tensor_data=torch.tensor(data) #데이터를 '텐서'로 담아주기(list->tensor)
print(f'일반 텐서: {tensor_data}')
#GPU가동 MacBook(mps)|window,geforce(cuda,cuddn)
if torch.backends.mps.is_available(): 
    #backends:하드웨어 가속을 위한 최적화 라이브러리(백엔드)의 설정을 제어
    mps_machine=torch.device('mps') #사용할 GPU:mps
    fast_machine=tensor_data.to(mps_machine) #CPU에 있었던 tensor_data를 .to()를 이용해 작업 영역을 GPU로 옮김
    print('GPU모드로 전환되었습니다.')
    print(f'mps tensor: {fast_machine}')
else: #GPU 경로가 차단된 경우
    print('CPU모드로 전환되었습니다.')

#device='mps:0':0번째 그래픽카드(M1칩 하나만 있기에)
#to.(): 작업 영역 옮기기