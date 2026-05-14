import torch
#utils:다용도 도구함|data:데이터의 형태 정의(중간 다리 역할)
#Dataset:데이터의 형태와 규칙을 정의하는 설계도|DataLoader:데이터를 효율적으로 옮기는 기능(컨베이어 벨트)
from torch.utils.data import Dataset,DataLoader
#환자 명부 시스템 구축
class EMRDataset(Dataset):#class:데이터셋 고유의 데이터(X,Y)와 이를 다루는 기능(__init__)을 하나의 객체(묶음)로 포장
    def __init__(self):#__init__(initialize):초기화 함수,클래스를 실행할 때,딱 한 번 실행됨
        self.patient_x=torch.randn(100,2)#행:100,열:2,무작위로 추출하여 텐서 생성(자세한 설명은 아래)
        self.patient_y=torch.randint(0,2,(100,1),dtype=torch.float)#0이상2미만 범위에서 무작위로 정수를 뽑아서 100행 1열을 만듦(정답지)
    def __len__(self): #길이(환자의 수가 얼마나 되나?)
        return len(self.patient_x)
    def __getitem__(self,idx):#idx(index):늘 다루던 것처럼 [0][1][2]... 구조가 똑같음
        return self.patient_x[idx],self.patient_y[idx]#patient_x[1]->1번 환자(행) 정보(열) 가져오기!
#DataLoader(컨베이어 벨트)|
emr_data=EMRDataset()#수백개의 정보를 GPU를 보낼시 메모리 제한으로 인해 터질 수 있어서, 따로 나눠서 보내야함
emr_loader=DataLoader(emr_data,batch_size=10,shuffle=True) #자세한 설명은 아래
#진료실 시뮬레이션
for batch_x,batch_y in emr_loader:
    print(f'진료실에 들어온 환자 수: {len(batch_x)}명')
    print(f'환자의 정보(X) 크기 {batch_x.shape}')#torch.Size[10,2]로 결과가 나옴->10명의 환자들은 각각 2개의 정보를 가지고 있음
    break

#class EMR~(Dataset):Dataset클래스를 이용하여 데이터셋 규격을 그대로 상속받아서 사용자만의 EMR규격을 만들어감
#__init__|(표준정규분포:아래 개념)
#평균이 0(가장 많은 값들이 몰려있는 범위=0) 표준편차가 1인(평균(0)을 중심으로 1만큼 떨어져있음) 정규분포(평균에 값이 몰려있고 극단적인 값은 드물게 있음)안에서 
#무작위로 숫자를 뽑아 100행 2열의 텐서 생성(환자 1명당 정보 2개)
#x,y는 문제지와 정답지(즉,train)관계->x에서 랜덤으로 100행,1열 정보(True or False) 생성->0~2(2미만)에서 100행,2열에 무작위로 정보를 출력해 적용 (2개의 열이 기준->세로)
#__getitem__:특정 정보 가져오기,행렬구조이니 인덱스를 이용하여 사용자가 가져오고 싶은 정보를 불러오면된다
#DataLoader:자동 컨베이어 벨트 기능,데이터를 효율적으로 옮긴다,즉 메모리 제한 때문에 batch_size를 이용해 양을 조정하여 옮기면된다
#shuffle:딥러닝이 정상50,비정상50을 학습할 때 순서대로 학습하면 답을 도출할 때 문제가 생기며,이를 방지하기 위해 무작위로 섞어 성능을 향상시킨다