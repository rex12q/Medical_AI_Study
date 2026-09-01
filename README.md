#AI Study
#Machine Learning, Deep Learning, Natural Language Processing..ETC

-작성자: 나 자신

-기간: 2025년 12월 11일 목요일 오후 2:09 ~ 현재까지

-해당 프로젝트는 더미 의료 데이터를 가지고 진행되었으며, 실제 EMR 데이터를 무단으로 배포, 공유했을 경우 법적 책임을 져야 한다는 사실을 인지하고 있습니다.

-프로젝트에 사용된 모든 모델은 의학적 진단 도구가 아닙니다.

-주요 프로젝트
무채혈 당뇨 스크리닝: 혈액 검사 없이 신체 계측치(BMI, 허리둘레, 혈압)만으로 당뇨 위험도 예측, 데이터 누수 방지를 위한 진단 기준 변수(공복혈당, 당화혈색소)를 피처에서 제외하고 설계

대사증후군 환자 군집화: K-Means로 환자를 자동 분류, 진단 기준과 얼마나 일치하는 지 검증. AI가 임상 기준을 재현할 수 있는 지에 대한 실험

SHAP(XAI) 기반 발병 원인 분석: XGBoost 예측 결과를 SHAP으로 해석, 어떤 지표가 대사증후군 판정에 기여했는지 시각화로 표현

StratifiedKFold, ROC_AUC, PR_AUC를 이용해 클래스 비율 확인: 층화된 KFold를 이용하여 올바르게 데이터를 학습할 수 있도록 설계, ROC_AUC, PR_AUC를 이용하여 불균형 데이터인지 확인 및 모델 성능 평가(상세 내용은 Classification Report로 확인함) 

-주로 다루는 내용: 
Module (Python)
numpy, pandas, matplotlib, seaborn 시각화 자료, pyreadstat 

Machine Learning (Python, Scikit Learn, XGB)
LogisticRegression, DecisionTree, Support Vector Machine, Voting, RandomForest, metrics(분류,회귀에 따른 지표 공부), preprocessing, model_selection(상세 파라미터 튜닝, 데이터 분할), Pipeline, Compose(열 합치기), SimpleImputer, XAI(SHAP), Cluster(군집화, 비지도학습)

Deep Learning (python, torch, tensorflow)
Neural Network, tensor(데이터 담기), optim, Stochastic Gradient Descent, backward, Dataset(파이프라인 설계), DataLoader(효율적으로 데이터 옮기기), Relu, Sigmoid, Linear, callbacks, Dense, Dropout, metrics(분류, 회귀에 따른 지표 공부)

Natural Language Processing (예제 파일은 DL 저장소에 있음)
CountVectorizer, TfidVectorizer (텍스트,숫자 벡터 변환)

-데이터 처리 및 활용
SPSS .sav 파일을 이용하여 메타데이터 추출  
임상 주제 기준에 기반한 모델 설계
결측 데이터 정리 및 불균형 처리 과정 

-이 저장소에 대하여
처음부터 학습과정을 정하지 않고 가장 기본이 되는 지식부터 시작을 했습니다. 혼자 LLM 툴을 이용하여 공부를 한 기록을 뒤늦게 정리하여 올렸기에 위에 작성한 타임라인과 맞지 않습니다. 초기 예제와 최근 예제 사이에는 클래스 설계, 난이도 수준의 차이가 분명히 있습니다.
그 차이를 삭제하지 않고 그대로 두었습니다. 데이터 누수를 모른 채 짠 코드도, 나중에 깨닫고 다시 설계한 코드도 함께 있습니다. 성능에 대한 평가와 왜 그렇게 흘러가는지 주석으로 남기는 것에 신경을 더 썼습니다. 이해하고 넘어가야 앞으로의 과정도 진행할 수 있기 때문입니다. 
배포가 가능한 완성된 프로젝트가 아닙니다. 학습에 대한 프로젝트가 맞습니다. 긴 글 읽어주셔서 감사합니다.
