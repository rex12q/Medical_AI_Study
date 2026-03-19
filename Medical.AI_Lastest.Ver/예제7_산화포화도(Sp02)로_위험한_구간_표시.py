#ML(Machine Learning)없이 시각적으로 나타낸 자료들도 중요하다는 걸 보여주는 예제
print('산화포화도(Sp02)로 위험한 구간 표시하기 (시각적 자료 포함)')
#matpltolib은 파이썬 한정 그래프 그리는 코드/pyplot 도구상자 (제목,선,점,보여주기..등)
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm #한글 폰트 지원
import matplotlib as mpl 

#mpl.rcParamas: 그래프 그릴 때 약속들을 의미한다.
mpl.rcParams['font.family'] = 'AppleGothic' # 애플고틱 이용
mpl.rcParams['axes.unicode_minus'] = False # -기호 깨지지 않게 true->false

#산화포화도(Sp02)
sp02=[98, 97, 96, 95, 93, 92, 94, 96, 97]

#한계점 변수
threshold=94

#데이터 나타내기 
print(f'산화포화도(sp02)의 데이터: {sp02}')

#그래프 나타내기
plt.plot(sp02, marker='o')#산화포화도에 o을 찍어 표현
plt.plot(threshold, linestyle='--')#94를 기준으로 점선을 표현
plt.title('산화포화도(sp02) 변화 (기준:94)')#제목
plt.xlabel('시간(또는 측정 순서)')#그래프니깐 x,y 제목 나타내기
plt.ylabel('sp02 (%)')
plt.show()#시각적 표현 코드

#위험 카운트
danger_count = 0 #0에서 시작
for v in sp02: # sp02안에 값들한테 v를 붙힘
    if v < threshold:
        danger_count += 1

print(f'주의 구간<{threshold}, 측정 횟수: {danger_count}') 


#Sp02(산화포화도)는 내 피 속에 산소가 얼마나 충분히 들었는지를 백분율로 나타낸 수치
#의미: 혈액 속의 산소 운반치(헤모글로빈)둘이 산소를 얼마나 가득 채우고 있는 지를 보여주는 지표
#수치(단위 퍼센트): 정상(95~100),주의(91~94),위험(90이하)
