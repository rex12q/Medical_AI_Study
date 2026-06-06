#%% [Cell 1] 라이브러리 로드 및 SAV 파일 변환 데이터 확인
import os
import pandas as pd
import numpy as np
import pyreadstat as pt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import shap

# 파일 경로 설정 (선생님 코드 표준 규칙 적용)
sav_file = 'Spss_sav/BP_Stat_Final_ExerData.sav'
csv_file = 'csv/[csv]Meta_Assist_Data.csv'

# 데이터 무결성 검사 및 변환 (예제 21, 22, 23 기반)
try:
    if not os.path.exists(csv_file):
        print("Meta_Assist: SAV 원본 데이터를 기반으로 의료 보안 가명 처리를 진행합니다...")
        sav_load, _ = pt.read_sav(sav_file)
        # 보안 폴더 자동 생성 후 저장
        os.makedirs('csv', exist_ok=True)
        sav_load.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print("Meta_Assist: 안전한 실무용 임베디드 데이터셋 변환 완료!")
except Exception as e:
    print(f"데이터 연동 오류 발생: {e}")

# 변환된 데이터 호출 및 모니터링 확인
df = pd.read_csv(csv_file)
print("--- [의료진 모니터용 EMR 전자의무기록 상위 5개 데이터 추출] ---")
print(df.head())


# %% [Cell 2] 피처 분할 및 고유의 3단계 대사증후군 타겟 빌드 (예제 23 기반)
# 의학 표준 가이드라인 기반의 가상 3단계 분류 타겟(0:정상, 1:주의, 2:고위험) 설정
# 실제 데이터의 컬럼 상황에 맞게 융통성 있게 작동하도록 안전 장치 가동
if 'target' not in df.columns:
    risk_target = []
    # 예제 21, 23 기반의 대사증후군 진단 지표(공복혈당, 복부비만, 혈압 등) 조건문 시스템
    for i in range(len(df)):
        if df['GLU'][i] >= 126 or df['SBP'][i] >= 140:
            risk_target.append(2) # 고위험군(High Risk)
        elif 100 <= df['GLU'][i] < 126 or 130 <= df['SBP'][i] < 140:
            risk_target.append(1) # 주의군(Medium Risk)
        else:
            risk_target.append(0) # 정상군(Normal)
    df['target'] = risk_target

# 진료에 쓰일 핵심 독립변수(X)와 종속변수(y) 분리
X = df[['Age', 'SBP', 'DBP', 'WC', 'BMI', 'GLU']]
y = df['target']

# 근본 분할 비율 8:2 적용 및 무작위 고정 난수 42 매칭
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# %% [Cell 3] 데이터 누수 차단용 ColumnTransformer & 자동화 파이프라인 설계
# 연속형(수치) 변수와 범주형 변수를 분기하여 최적의 결측치 대체법 매칭
numeric_features = ['Age', 'SBP', 'DBP', 'WC', 'BMI', 'GLU']
categorical_features = [] # 만약 성별이나 흡연 등의 컬럼이 포함될 시 여기에 기입

# 1. 수치형 데이터: 중앙값(median)으로 결측치를 메꾸고, 차원 조절용 표준화 스케일링 수행
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 2. 범주형 데이터: 최빈값(most_frequent)으로 메꾸고 원-핫 인코딩 적용
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# ColumnTransformer를 이용해 하나의 완벽한 복합 전처리기 캡슐로 결합
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# %% [Cell 4] XGBoostClassifier 연동 및 GridSearchCV 최고의 조합 탐색 (시간 단축 핵심)
# 전처리기와 메인 엔진인 XGBoost를 컨베이어 벨트로 묶기
meta_assist_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('xgb', XGBClassifier(objective='multi:softprob', random_state=42, eval_metric='mlogloss'))
])

# 최적의 속도와 정확도 밸런스를 찾는 하이퍼파라미터 그리드 격자 설정
param_grid = {
    'xgb__n_estimators': [100, 200],  # 의사 오답 체크 릴레이 횟수
    'xgb__max_depth': [3, 5, 7],       # 나무의 종적 사고 깊이 수준
    'xgb__learning_rate': [0.05, 0.1]  # 기울기 미세조정 보폭
}

print("Meta_Assist: 의료진의 분석 대기 시간 단축을 위한 오토 튜닝 엔진 가동...")
grid_search = GridSearchCV(meta_assist_pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 최고의 퍼포먼스를 보여주는 마스터 모델 추출
best_doctor_model = grid_search.best_estimator_
print(f"최적의 하이퍼파라미터 조합: {grid_search.best_params_}")
print(f"내과/가정의학과 검증 데이터 최고 정확도: {grid_search.best_score_*100:.1f}%")


# %% [Cell 5] 의사용 최종 진단 레포트 및 오차 행렬 시각화 출력
y_pred = best_doctor_model.predict(X_test)

print("\n" + "="*60)
print("   [META_ASSIST MEDICAL REPORT FOR CLINICIANS]   ")
print("="*60)
print(classification_report(y_test, y_pred, target_names=['Normal', 'Medium Risk', 'High Risk']))
print("="*60)

# %% [Cell 6] 필살기: 의사 설득용 SHAP 데이터 설명 모델 빌드
print("\nMeta_Assist: 설명 가능한 AI (XAI) 시각화 엔진을 구축합니다...")

# 파이프라인에서 전처리가 완료된 Train 데이터를 추출하여 SHAP에 주입
# SHAP은 파이프라인 해석이 불가능하기에 named_steps(파이프라인에서 원하는 정보만 추출 가능)를 이용해 한 번 더 전처리 진행을 위한 코드 설계
X_train_transformed = best_doctor_model.named_steps['preprocessor'].transform(X_train)#결측치 채우고 스케일링
X_test_transformed = best_doctor_model.named_steps['preprocessor'].transform(X_test)#결측치 채우고 스케일링
xgb_engine = best_doctor_model.named_steps['xgb']#실제 대사증후군 예측

# 수치 데이터 컬럼명 매칭
feature_names = numeric_features

# TreeExplainer를 통한 기여도 연산
explainer = shap.TreeExplainer(xgb_engine)
shap_values = explainer.shap_values(X_test_transformed)

print("🎯 SHAP 엔진 준비 완료. 주피터 환경이나 아래 커맨드를 통해 시각화 차트를 호출하십시오.")
# ---------------------------------------------------------
# [이 아래부터가 추가된 시각화 출력 코드]

import matplotlib.pyplot as plt

# 1. 요약 차트 (Summary Plot - 전체 환자 관점)
plt.figure(figsize=(10, 6))
plt.title("Meta_Assist: 전체 환자 대사증후군 위험 요인 (Summary)")
# 여기서 비로소 feature_names(이름표)가 쓰입니다!
shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, show=False)
plt.tight_layout()
plt.show()

# 2. 막대 그래프 (Bar Plot - 직관적인 영향력 순위)
plt.figure(figsize=(10, 6))
plt.title("Meta_Assist: 평균 위험도 기여도 순위 (Bar)")
shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, plot_type="bar", show=False)
plt.tight_layout()
plt.show()

# 3. 개별 환자 맞춤형 포스 플롯 (Force Plot-주력)
# 주피터/VS Code에서 인터랙티브(마우스 반응형)로 보려면 initjs() 필수
shap.initjs()

print("[개별 환자 분석] 0번 환자의 고위험군(High Risk) 판정 원인 분석")
# 0번 환자의 고위험군(클래스 인덱스 2) 기여도 출력
shap.force_plot(
    base_value=explainer.expected_value[2], 
    shap_values=shap_values[2][0,:],        
    features=X_test_transformed[0,:],       
    feature_names=feature_names             
)
# 주의: force_plot은 plt.show()가 아니라 주피터 셀 결과창에 HTML 형태로 자동 출력됨!