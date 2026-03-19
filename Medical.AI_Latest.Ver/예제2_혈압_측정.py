#systolic: 수축기(sys) , diatolic: 이완기(dia), blood pressure(bp)
print('혈압을 측정해 봅시다 (예제용)')
print()

sys_bp=int(input('수축기 혈압 숫자 입력:'))
dia_bp=int(input('이완기 혈압 숫자 입력:'))

def classify_bp(sys_bp, dia_bp): #def(함수), return은 세트/ def 안에서 계산한 값을 밖으로 전달하려면 retrun
    if sys_bp < 120 and dia_bp < 80:
        return '정상 혈압'
    elif sys_bp < 139 and dia_bp < 89:
        return '고혈압 전 단계. 주의!'
    elif sys_bp < 159 and dia_bp < 99:
        return '1기 (경도)고혈압. 당장 조치 필요!'
    else:
        return '2기 (중등도 이상)고혈압. 당장 조치 필요!'

result=classify_bp(sys_bp, dia_bp) #값을 받기 위해 result라는 변수에 함수를 담음
print(f'결과: {result}') #format을 앞에 두고 선언 (f)

#출처 서울아산병원, 미국심장학회의(혈합 기준 자료) 
#혈압: 동맥혈관 벽에 대항한 혈핵의 압력
#심장이 수축하여 동맥혈관으로 혈액을 보낼 때 가장 높은데, 이걸 수축기 혈압/ 심장이 늘어나서 받아들일 때 가장 낮은데, 이걸 이완기 혈압 