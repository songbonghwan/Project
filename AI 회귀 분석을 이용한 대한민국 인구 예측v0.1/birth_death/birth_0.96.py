import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score

# 데이터 불러오기 (2007년부터 2023년까지의 데이터)
house_price = pd.read_csv('csv/집값(평균)_full.csv', index_col='AREA')
feat_sal = pd.read_csv('csv/feat_sal.csv', index_col='AREA') 
education_expense = pd.read_csv('csv/사교육비 총액_full.csv', index_col='AREA')
birth_rate = pd.read_csv('csv/시군구_출생아수_full.csv', index_col='AREA')
fertile_women = pd.read_csv('csv/가임 여성수_full.csv', index_col='AREA')
marriage_count = pd.read_csv('csv/혼인건수_full.csv', index_col='AREA')

# 데이터 전처리
def clean_comma_columns(df):
    for column in df.columns:
        df[column] = df[column].replace({',': ''}, regex=True).astype(float)
    return df

house_price = clean_comma_columns(house_price)
education_expense = clean_comma_columns(education_expense)
birth_rate = clean_comma_columns(birth_rate)
fertile_women = clean_comma_columns(fertile_women)
marriage_count = clean_comma_columns(marriage_count)

# 데이터의 공통된 연도 선택
house_price.columns = house_price.columns.str.strip()
feat_sal.columns = feat_sal.columns.str.strip() 
education_expense.columns = education_expense.columns.str.strip()
birth_rate.columns = birth_rate.columns.str.strip()
fertile_women.columns = fertile_women.columns.str.strip()
marriage_count.columns = marriage_count.columns.str.strip()

common_years = list(set(house_price.columns) & set(feat_sal.columns) & set(education_expense.columns) & 
                    set(birth_rate.columns) & set(fertile_women.columns) & set(marriage_count.columns))

if len(common_years) == 0:
    raise ValueError("공통된 연도가 없습니다. 데이터 확인이 필요합니다.")

# 데이터를 공통 연도로 필터링
house_price = house_price[common_years]
feat_sal = feat_sal[common_years]  # 임금 데이터를 feat_sal로 변경
education_expense = education_expense[common_years]
birth_rate = birth_rate[common_years]
fertile_women = fertile_women[common_years]
marriage_count = marriage_count[common_years]

# 1. 지역별로 예측을 진행할 수 있도록 반복문을 추가
# 모든 지역에 대해 모델을 훈련하고 예측
regions = house_price.index  # 지역 목록

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
        'house_price': house_price.loc[region], 
        'wage': feat_sal.loc[region], 
        'education_expense': education_expense.loc[region],
        'fertile_women': fertile_women.loc[region],
        'marriage_count': marriage_count.loc[region]
    })

    y = birth_rate.loc[region]

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
        'house_price': house_price.loc[region, years_to_predict.astype(str)], 
        'wage': feat_sal.loc[region, years_to_predict.astype(str)],  
        'education_expense': education_expense.loc[region, years_to_predict.astype(str)],
        'fertile_women': fertile_women.loc[region, years_to_predict.astype(str)],
        'marriage_count': marriage_count.loc[region, years_to_predict.astype(str)]
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
print("\n2007년부터 2023년까지 각 지역별 예측된 출생아수:")
for region, pred in predictions.items():
    print(f"\n지역: {region}")
    for year, birth_count in zip(years_to_predict, pred):
        print(f"{year}년 예측 출생아수: {birth_count:.2f}")

# -------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# 예측된 2024년부터 2075년까지의 데이터 불러오기
house_price_2024_2075 = pd.read_csv('Sarimax_csv/예측된_집값_2024_2075.csv')
feat_sal_2024_2075 = pd.read_csv('Sarimax_csv/예측된_임금_총액_2024_2075.csv')
education_expense_2024_2075 = pd.read_csv('Sarimax_csv/예측된_사교육비_총액_2024_2075.csv')
fertile_women_2024_2075 = pd.read_csv('Sarimax_csv/예측된_가임_여성수_2024_2075.csv')
marriage_count_2024_2075 = pd.read_csv('Sarimax_csv/예측된_혼인건수_2024_2075.csv')

# 공통된 연도 (2024년부터 2075년까지)
years_to_predict_future = np.arange(2024, 2076)

# 공통적인 도시 목록
cities = house_price_2024_2075['City'].values

# 결과를 저장할 빈 데이터프레임 초기화
predictions_df = pd.DataFrame(columns=['City'] + years_to_predict_future.astype(str).tolist())

# 기존 학습된 모델을 사용하여 각 도시별로 예측을 수행
future_predictions = {}

for city in cities:
    print(f"예측을 위한 도시: {city}")
    
    # 기존 데이터를 사용하여 해당 도시의 학습된 모델을 불러옵니다.
    # 특성 행렬(X)와 목표 변수 벡터(y) 설정 (지역별로)
    X = pd.DataFrame({
        'house_price': house_price.loc[city], 
        'wage': feat_sal.loc[city], 
        'education_expense': education_expense.loc[city],
        'fertile_women': fertile_women.loc[city],
        'marriage_count': marriage_count.loc[city]
    })
    
    y = birth_rate.loc[city]

    # 데이터 표준화 -> MinMaxScaler로 변경하여 변화폭을 줄임
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # PolynomialFeatures 차수 감소 (degree=1로 변경하여 모델 복잡도를 낮춤)
    poly = PolynomialFeatures(degree=1, include_bias=False)  # degree=1로 변경
    X_poly = poly.fit_transform(X_scaled)

    # Ridge 모델 하이퍼파라미터 튜닝 (GridSearchCV)
    param_grid = {
        'alpha': np.logspace(-4, 1, 6)  # alpha 값을 조금 더 크게 설정하여 규제 강화를 시도
    }

    grid_search = GridSearchCV(Ridge(max_iter=10000, random_state=42), param_grid, cv=5, scoring='r2')
    grid_search.fit(X_poly, y)

    # 훈련된 모델을 사용하여 2024-2075년 예측
    best_model = grid_search.best_estimator_

    # 예측할 미래 데이터 준비
    X_future = pd.DataFrame({
        'house_price': house_price_2024_2075.loc[house_price_2024_2075['City'] == city, years_to_predict_future.astype(str)].values.flatten(),
        'wage': feat_sal_2024_2075.loc[feat_sal_2024_2075['City'] == city, years_to_predict_future.astype(str)].values.flatten(),
        'education_expense': education_expense_2024_2075.loc[education_expense_2024_2075['City'] == city, years_to_predict_future.astype(str)].values.flatten(),
        'fertile_women': fertile_women_2024_2075.loc[fertile_women_2024_2075['City'] == city, years_to_predict_future.astype(str)].values.flatten(),
        'marriage_count': marriage_count_2024_2075.loc[marriage_count_2024_2075['City'] == city, years_to_predict_future.astype(str)].values.flatten()
    })

    # X_future 데이터 표준화
    X_future_scaled = scaler.transform(X_future)

    # PolynomialFeatures 적용 (훈련 시 사용한 것과 동일하게 변환)
    X_future_poly = poly.transform(X_future_scaled)

    # 예측값 계산
    future_predictions_city = best_model.predict(X_future_poly)
    
    # 예측값이 0보다 작은 경우는 0으로 설정
    future_predictions_city = np.maximum(future_predictions_city, 0)  # 출생아수는 0보다 커야 하므로 0으로 설정

    # 예측값의 변화폭을 제한 (급격한 감소를 방지하기 위해)
    for i in range(1, len(future_predictions_city)):
        max_decrease = future_predictions_city[i-1] * 0.96  # 3% 이하로 감소하도록 제한
        if future_predictions_city[i] < max_decrease:
            future_predictions_city[i] = max_decrease
    
    # 결과 저장
    future_predictions[city] = future_predictions_city

    # 예측된 신생아수 데이터를 DataFrame으로 변환하여 predictions_df에 추가
    predictions_df.loc[len(predictions_df)] = [city] + future_predictions_city.tolist()

# 예측된 데이터를 '예측된_신생아수_2024_2075.csv' 파일로 저장
predictions_df.to_csv('예측된_신생아수_2024_2075.csv', index=False)

# 결과 출력
print("\n2024년부터 2075년까지 각 도시별 예측된 출생아수가 '예측된_신생아수_2024_2075.csv'에 저장되었습니다.")
