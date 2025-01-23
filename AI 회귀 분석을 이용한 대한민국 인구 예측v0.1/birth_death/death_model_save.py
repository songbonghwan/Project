import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
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
regions = cancer_data.index  # 지역 목록

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

    # 3. PolynomialFeatures 적용
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)

    # 4. 훈련 데이터와 테스트 데이터로 분리
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

    # 5. Ridge 모델 하이퍼파라미터 튜닝 (GridSearchCV)
    param_grid = {
        'alpha': np.logspace(-4, 1, 6)
    }

    grid_search = GridSearchCV(Ridge(max_iter=10000, random_state=42), param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

    best_alpha = grid_search.best_params_['alpha']
    print(f"Best alpha for Ridge in {region}: {best_alpha}")

    best_model = grid_search.best_estimator_

    # 훈련 점수와 테스트 점수
    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)

    print(f"Train score with best model for {region}: {train_score}")
    print(f"Test score with best model for {region}: {test_score}")

    # 예측값 계산
    y_pred = best_model.predict(X_test)

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

    # PolynomialFeatures 적용 (훈련 시 사용한 것과 동일하게 변환)
    X_future_poly = poly.transform(X_future_scaled)

    # 예측값 계산
    future_predictions = best_model.predict(X_future_poly)

    # 예측된 값 저장
    predictions[region] = future_predictions

# 결과 출력
print("\n2007년부터 2023년까지 각 지역별 예측된 사망자수:")
for region, pred in predictions.items():
    print(f"\n지역: {region}")
    for year, death_count in zip(years_to_predict, pred):
        print(f"{year}년 예측 사망자수: {death_count:.2f}")

import joblib

# 모델 저장 (각 지역에 대해)
for region in regions:
    # 학습된 모델 저장
    joblib.dump(best_model, f'model_{region}.pkl')


# ------------------------------------------------------------------------------------------------------------------------------------


# 1. 예측된 2024년부터 2075년까지의 데이터 불러오기
elderly_population_data = pd.read_csv('Sarimax_csv/예측된_노인_인구_2024_2075.csv')
cancer_data = pd.read_csv('Sarimax_csv/예측된_암환자_2024_2075.csv')
cyc_disease_data = pd.read_csv('Sarimax_csv/예측된_순환계질환_사망건수_2024_2075.csv')
mental_disease_data = pd.read_csv('Sarimax_csv/예측된_정신질환_사망건수_2024_2075.csv')

# 2. 공통된 연도 (2024년부터 2075년까지)
years_to_predict_future = np.arange(2024, 2076)

# 3. 공통적인 도시 목록 (여기서는 'City' 컬럼이 존재한다고 가정)
cities = elderly_population_data['City'].values

# 4. 결과를 저장할 빈 데이터프레임 초기화
predictions_df = pd.DataFrame(columns=['City'] + years_to_predict_future.astype(str).tolist())

# 5. 기존 학습된 모델을 사용하여 각 도시별로 예측을 수행
future_predictions = {}

# 'death_data'에 존재하는 연도만 예측에 사용
common_years_in_death_data = np.array([str(year) for year in range(2007, 2024)])

# 6. 각 도시별로 예측을 진행
for city in cities:
    print(f"예측을 위한 도시: {city}")
    
    # 특성 행렬(X)와 목표 변수 벡터(y) 설정 (지역별로)
    # 인덱스를 'City'로 설정
    elderly_population_data.set_index('City', inplace=True)
    cancer_data.set_index('City', inplace=True)
    cyc_disease_data.set_index('City', inplace=True)
    mental_disease_data.set_index('City', inplace=True)
    
    # 이제 city 값으로 각 데이터를 참조
    X = pd.DataFrame({
        'elderly_population': elderly_population_data.loc[city].values,
        'cancer_data': cancer_data.loc[city].values,
        'cyc_disease': cyc_disease_data.loc[city].values,
        'mental_disease': mental_disease_data.loc[city].values
    })

    # y 설정을 각 연도별로 맞춰줍니다.
    y = death_data.loc[city, common_years_in_death_data].values  # 'death_data'에 존재하는 연도만 사용

    # 데이터 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PolynomialFeatures 적용 (degree=2로 설정)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)

    # 모델 불러오기
    try:
        best_model = joblib.load(f'model_{city}.pkl')
    except FileNotFoundError:
        print(f"모델 파일을 찾을 수 없습니다: model_{city}.pkl")
        continue

    # 훈련된 모델을 사용하여 2024-2075년 예측
    X_future = pd.DataFrame({
        'elderly_population': elderly_population_data.loc[city, years_to_predict_future.astype(str)].values.flatten(),
        'cancer_data': cancer_data.loc[city, years_to_predict_future.astype(str)].values.flatten(),
        'cyc_disease': cyc_disease_data.loc[city, years_to_predict_future.astype(str)].values.flatten(),
        'mental_disease': mental_disease_data.loc[city, years_to_predict_future.astype(str)].values.flatten()
    })

    # X_future 데이터 표준화
    X_future_scaled = scaler.transform(X_future)

    # PolynomialFeatures 적용 (훈련 시 사용한 것과 동일하게 변환)
    X_future_poly = poly.transform(X_future_scaled)

    # 예측값 계산
    future_predictions_city = best_model.predict(X_future_poly)
    
    # 예측값이 0보다 작은 경우는 0으로 설정
    future_predictions_city = np.maximum(future_predictions_city, 0)  # 사망자수는 0보다 커야 하므로 0으로 설정
    
    # 급격한 변화가 없도록 예측값의 변화를 제한
    future_predictions_city = np.clip(future_predictions_city, future_predictions_city.min(), future_predictions_city.max())  # 예측값이 급격히 변하지 않도록 제한
    
    # 결과 저장
    future_predictions[city] = future_predictions_city

    # 예측된 사망자수를 DataFrame으로 변환하여 predictions_df에 추가
    predictions_df.loc[len(predictions_df)] = [city] + future_predictions_city.tolist()

# 7. 예측된 데이터를 '예측된_사망자수_2024_2075.csv' 파일로 저장
predictions_df.to_csv('예측된_사망자수_2024_2075.csv', index=False)

# 8. 결과 출력
print("\n2024년부터 2075년까지 각 도시별 예측된 사망자수가 '예측된_사망자수_2024_2075.csv' 파일에 저장되었습니다.")
