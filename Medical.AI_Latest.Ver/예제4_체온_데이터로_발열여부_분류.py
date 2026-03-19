print('체온 데이터로 발열 여부를 분류. (그래프 버전)')
#matplotlib은 그래프 그리는 라이브러리 (파이썬 한정)/pyplot=수많은 기능들이 내재된 도구상자
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm # 한글 폰트 지원
import matplotlib as mpl # mpl로 줄임

#mpl.rcParams: 그래플 그릴 때 약속들을 의미
mpl.rcParams['font.family'] = 'AppleGothic' # 애플고틱 이용
mpl.rcParams['axes.unicode_minus'] = False # -기호 깨지지 않게 true->false

temps = [
    36.5, 
    36.8, 
    37.0, 
    37.3, 
    37.6, 
    38.0, 
    38.5, 
    36.9, 
    37.2
    ]

labels = [] #체온이 정상인지 발열인지 기록할 공간 만들기
for t in temps: #temps에서 t라는 값을 대입해 꺼낸다
    if t >= 37.5:
        labels.append("발열") #37.5가 이상이면 labels에 "발열"추가
    else:
        labels.append("정상") #아니면 "정상"추가

print(f'체온 결과 리스트:{temps}')
print(f'분류 결과:{labels}')

plt.plot(temps, marker='o') #temps 안에 있는 체온들을 0,1,2..번 순서로 점(marker)을 찍고 그 점(marker)들을 선으로 이어야 함
plt.axhline(37.5, linestyle='--') #37.5 선을 표현함 (시각적 표현 코드)
plt.title('시각적 자료 (발열 기준:37.5)') #제목
plt.xlabel('사람 번호') #x:사람번호
plt.ylabel('체온') #y:체온
plt.show()#시각적 표현 코드