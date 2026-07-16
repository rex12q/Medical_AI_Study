import torch
import torch.nn as nn
import torch.optim as optim
#Term Frequency(TF): 텍스트에 특정 단어가 얼마나 존재
# Inverse Document Frequency(IDE): 다른 텍스트(문서)에서도 흔하게 나오는 단어인지 확인(만약 비중이 많을 시 감점 요소 적용) 
from sklearn.feature_extraction.text import TfidfVectorizer

#1. Text Info (0:NEG,1:POS)
#텍스트를 Matrix Shape로 바라보자! 
h_text = [
    "Severe chest pain, needs immediate oxygen.", # 1 (응급) 가슴 통증, 즉시 산소 필요
    "Patient needs a routine checkup.",           # 0 (비응급) 환자 루틴 확인 필요
    "Sudden cardiac arrest, starting CPR.",       # 1 (응급) 갑작스러운 심정지 CPR 시작
    "Slight fever for two days."                  # 0 (비응급) 이틀동안 미열 
] #x
labels=[1,0,1,0] #y

vectorizer=TfidfVectorizer() #모듈 생성
X_tfidf=vectorizer.fit_transform(h_text).toarray() # 파이토치에 넣기 위한 배열 작업

#Word Dict 개수 파악 (=신경망 입력 크기)
input_size=X_tfidf.shape[1] #[0]:row, [1]:col (단어 개수)

#Convert pytorch -> tensor (과정 같음)
X_tensor = torch.tensor(X_tfidf, dtype=torch.float)
y_tensor= torch.tensor(labels, dtype=torch.float).view(-1,1) #행: 전체 load, col: 1

#DL 모델 설계 (Word Dict 참고)
class DL_NLP(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.nlp_model=nn.Sequential(
            nn.Linear(input_features,16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16,1),
            nn.Sigmoid() #이진 분류 1,0
        )
    def forward(self, x):
        return self.nlp_model(x)
    
#모듈 소환 및 테스트
doctor=DL_NLP(input_features=input_size) #input_size
bce=nn.BCELoss()
optimizer=optim.Adam(doctor.parameters(),lr=0.01)

#epoch 반복학습
Epochs=1000
for epoch in range(Epochs):
    studying=doctor(X_tensor)
    loss=bce(studying,y_tensor) #정답지와 함께 비교
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 초기 상태에서 텐서 통과
    doctor.eval()
    with torch.no_grad():
        pred=doctor(X_tensor) 
    if (epoch+1) % 10 == 0:
        print(f'Epoch: [{epoch+1}/{Epochs}] | Loss: [{loss.item():.4f}]')

print('결과 출력')
print(pred)
