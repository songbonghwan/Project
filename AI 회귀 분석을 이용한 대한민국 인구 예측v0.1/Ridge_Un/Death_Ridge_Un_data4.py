import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# 새로운 데이터 불러오기
death_data = pd.read_csv('csv/사망자수_ADD_full.csv', index_col='AREA')
elderly_population_data = pd.read_csv('csv/노인 인구_full.csv', index_col='AREA')
cancer_data = pd.read_csv('csv/암 등록환자현황_full.csv', index_col='AREA')
feat_cyc_data = pd.read_csv('csv/순환계질환cyc.csv', index_col='AREA')
feat_ment_data = pd.read_csv('csv/정신질환ment.csv', index_col='AREA')

# 데이터를 전처리
def clean_comma_columns(df):
    for column in df.columns:
        df[column] = df[column].replace({',': ''}, regex=True).astype(float)
    return df

death_data = clean_comma_columns(death_data)
elderly_population_data = clean_comma_columns(elderly_population_data)
cancer_data = clean_comma_columns(cancer_data)
feat_cyc_data = clean_comma_columns(feat_cyc_data)
feat_ment_data = clean_comma_columns(feat_ment_data)

# 데이터의 공통된 연도 선택
death_data.columns = death_data.columns.str.strip()
elderly_population_data.columns = elderly_population_data.columns.str.strip()
cancer_data.columns = cancer_data.columns.str.strip()
feat_cyc_data.columns = feat_cyc_data.columns.str.strip()
feat_ment_data.columns = feat_ment_data.columns.str.strip()

common_years = list(set(death_data.columns) & set(elderly_population_data.columns) & 
                    set(cancer_data.columns) & set(feat_cyc_data.columns) & 
                    set(feat_ment_data.columns))

if len(common_years) == 0:
    raise ValueError("공통된 연도가 없습니다. 데이터 확인이 필요합니다.")

# 데이터를 공통 연도로 필터링
death_data = death_data[common_years]
elderly_population_data = elderly_population_data[common_years]
cancer_data = cancer_data[common_years]
feat_cyc_data = feat_cyc_data[common_years]
feat_ment_data = feat_ment_data[common_years]

# 1. 지역별로 예측을 진행할 수 있도록 반복문을 추가
# 모든 지역에 대해 모델을 훈련하고 예측
regions = death_data.index  # 지역 목록

# 결과를 저장할 딕셔너리
predictions = {}

# 'ADD_UP' 지역은 제외
regions = regions[regions != 'ADD_UP']

for region in regions:
    # 지역 이름에 공백이 있을 경우 처리
    region = region.strip()

    print(f"예측을 위한 지역: {region}")

    # 특성 행렬(X)와 목표 변수 벡터(y) 설정 (지역별로)
    X = pd.DataFrame({
        'elderly_population': elderly_population_data.loc[region], 
        'cancer_data': cancer_data.loc[region],
        'cyc_disease': feat_cyc_data.loc[region],  # 순환계 질환 데이터 추가
        'mental_disease': feat_ment_data.loc[region]  # 정신 질환 데이터 추가
    })

    y = death_data.loc[region]

    # 2. 데이터 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. 훈련 데이터와 테스트 데이터로 분리
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 4. Ridge 모델 훈련
    model = Ridge(alpha=1.0, max_iter=10000, random_state=42)
    model.fit(X_train, y_train)

    # 훈련 점수와 테스트 점수
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"Train score with best model for {region}: {train_score}")
    print(f"Test score with best model for {region}: {test_score}")

    # 예측값 계산
    y_pred = model.predict(X_test)

    # R2 평가 지표 출력
    r2 = r2_score(y_test, y_pred)
    print(f"R-squared for {region}: {r2}")

    # 예측할 연도 배열 (2007년부터 2023년까지)
    years_to_predict = np.array([2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])

    # 각 연도에 대해 해당하는 특성값을 추출
    X_future = pd.DataFrame({
        'elderly_population': elderly_population_data.loc[region, years_to_predict.astype(str)], 
        'cancer_data': cancer_data.loc[region, years_to_predict.astype(str)],
        'cyc_disease': feat_cyc_data.loc[region, years_to_predict.astype(str)],  # 순환계 질환 데이터 추가
        'mental_disease': feat_ment_data.loc[region, years_to_predict.astype(str)]  # 정신 질환 데이터 추가
    })

    # X_future 데이터 표준화
    X_future_scaled = scaler.transform(X_future)

    # 예측값 계산
    future_predictions = model.predict(X_future_scaled)

    # 예측된 값 저장
    predictions[region] = future_predictions

# 결과 출력
print("\n2007년부터 2023년까지 각 지역별 예측된 사망자수:")
for region, pred in predictions.items():
    print(f"\n지역: {region}")
    for year, death_count in zip(years_to_predict, pred):
        print(f"{year}년 예측 사망자수: {death_count:.2f}")
