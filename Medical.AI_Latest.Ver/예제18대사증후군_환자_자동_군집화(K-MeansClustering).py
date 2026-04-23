#본 EMR은 한림대학교 '확률과 통계' 수업 시간 때 제공받은 병원 데이터를 가져온 것이므로 무단으로 유포하지 않을 것을 선서합니다.
#사용자는 EMR을 함부로 유포하면 국내 의료법에 의거하여 법적 처벌을 받을 수 있다는 것을 인지합니다.
#%%
import pandas as pd
import matplotlib.pyplot as plt
import os
import pyreadstat as pt
import seaborn as sns
from sklearn.cluster import KMeans
#cluster(군집화): 스펙(x)을 깔아놓고 서로 비슷한 스탯(거리가 가까움)끼리 묶어서 그룹을 형성하는 개념, 즉 비지도 학습이다
#k(군집수)_means: 군집화(클러스터링) 개념을 기반으로 가장 많이 사용되는 ML중 하나이며 하드 클러스터링 기법이다
#데이터를 k개의 클러스터(군집화)로 나누고 각 데이터 포인트를 중심에 배치한다 
from sklearn.pipeline import make_pipeline #결측치,클러스터 한 과정에 다 때려넣기!
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sav_load = 'Spss_sav/BP_Stat_Final_ExerData.sav'
sa_cs = 'csv/[csv]Metabolic_syndrome_Data.csv' 
#Metabolic_syndrome_Data.csv -> csv/[csv]이하동일 (실행할 때 마다 파일 덮어씌워짐)

if not os.path.exists(sa_cs):
    try:
        savRead,_ = pt.read_sav(sav_load) #metadata 빼기
        print('Converting...')
        savRead.to_csv(sa_cs,index=False,encoding='utf-8-sig')
        print('Complete!')
    except Exception as e:
        print(f'Oops! {e}')
csv_load = pd.read_csv(sa_cs)
#metadataonly
try:
    _,meta = pt.read_sav(sav_load,metadataonly=True) #metadata만 가져오기(알맹이)
    for c_name,c_label in meta.column_names_to_labels.items(): #열에 있는 이름과 라벨 가져오기(items)
        label = c_label if c_label else 'Not explanation'
        print(f'Explanation: {label}')
except Exception as e:
    print(f'ERROR: {e}')

print('Complete load csv file!') #성공!
#cluster and pipeline
X_info = csv_load[['BMI','WC','SBP','DBP','GLU']]
doctor_clu = make_pipeline(
    StandardScaler(),
    KMeans(n_clusters=3,random_state=42) #군집화 갯수 4로, 섞는 방법 42로 고정
)
#Cluster_col 생성
csv_load['Cluster_col'] = doctor_clu.fit_predict(X_info)
#fit_predict: 학습과 동시에 바로 예측을 하는 내장어
print('-'*50)
print('BMI,WC,GLU cluster result') #군집화 결과
print(csv_load[['BMI','WC','GLU']].head(3)) #상단 3개
print('-'*50)
print('X_info group mean') #각 그룹별(group by) 군집화 평균
cluster_mean = csv_load.groupby('Cluster_col')[['BMI','WC','SBP','DBP','GLU']].mean() #평균
#groupby는 기능(메서드)이기에 ()| pd는 불러오는 것이기에 []
print("Result")
for cluster_num,row in cluster_mean.iterrows(): #iter(반복)rows: 표 위에서부터 한 줄씩(row) 읽어 내려오는 명령어
    result_clu = f"[{cluster_num}. group mean] Body Mess Index: {row['BMI']:.1f}| Waist Circumstance: {row['WC']:.1f}| Systololic Blood Pressure: {row['SBP']:.1f}| Diatolic Blood Pressure: {row['DBP']:.1f}| Glucose: {row['GLU']:.1f}"
    #row[]열(칸)이 있는 곳에 행(row)의 내용을 가져오기(row,col 혼란x)| 5개의 그룹을 한 row로
    print(result_clu) #각 그룹 번호는 의미 없음, 그냥 무작위로 던져주는 번호(이를 방지하기 위해 random_state로 고정)
print('-'*50)
print('Cluster Visualize')
#군집화 그룹 이름 형성
groupName = '' 
for cluster_num,row in cluster_mean.iterrows():
    if row['BMI'] >= 30.0 or row['GLU'] >= 150:
        groupName = 'Extreme High Risk Group'
    elif 23.0 <= row['BMI'] < 30.0 or 100 <= row['GLU'] < 150:
        groupName = 'Mediumn Risk Group'
    else:
        groupName = 'Normal Group'
    print('-'*50)
    print(f'{cluster_num}. Result - Group: {groupName}')
    print(f'BMI: {row['BMI']:.1f} (Overweight > 23.0)')
    print(f'WC: {row['WC']:.1f} (Male > 90,Female > 85)')
    print(f'SBP: {row['SBP']:.1f} | DBP: {row['DBP']:.1f} (Hypertension: 140|90)')
    print(f'GLU: {row['GLU']:.1f} (Prediabetes > 100)')
#그래프 길이 설정
plt.figure(figsize=(8,6))
sns.scatterplot(x='BMI',y='GLU',hue='Cluster_col', data=csv_load)
#hue:(data 있다는 전제 하에)해당 열에 있는 정보들을 알아서 맞춰주고 안내표까지 그려주는 기능
#data=''가 이미 있기에 환경이 data로 바뀜
plt.title('BMI vs GLU')
plt.show()
print('END')
# %%
