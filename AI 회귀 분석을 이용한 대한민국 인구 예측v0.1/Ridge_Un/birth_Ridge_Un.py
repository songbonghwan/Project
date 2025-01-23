import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# 데이터 불러오기
house_price = pd.read_csv('csv/집값(평균)_full.csv', index_col='AREA')
feat_sal = pd.read_csv('csv/feat_sal.csv', index_col='AREA') 
education_expense = pd.read_csv('csv/사교육비 총액_full.csv', index_col='AREA')
birth_rate = pd.read_csv('csv/시군구_출생아수_full.csv', index_col='AREA')
fertile_women = pd.read_csv('csv/가임 여성수_full.csv', index_col='AREA')
marriage_count = pd.read_csv('csv/혼인건수_full.csv', index_col='AREA')

# 데이터를 전처리
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
feat_sal = feat_sal[common_years]
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

    # 3. 훈련 데이터와 테스트 데이터로 분리
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 4. Ridge 모델 훈련
    model = Ridge(alpha=1.0, max_iter=10000, random_state=42)  # alpha 기본값은 1.0
    model.fit(X_train, y_train)

    # 훈련 점수와 테스트 점수
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"Train score for {region}: {train_score}")
    print(f"Test score for {region}: {test_score}")

    # 예측값 계산
    y_pred = model.predict(X_test)

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

    # 예측값 계산
    future_predictions = model.predict(X_future_scaled)

    # 예측된 값 저장
    predictions[region] = future_predictions

# 결과 출력
print("\n2007년부터 2023년까지 각 지역별 예측된 출생아수:")
for region, pred in predictions.items():
    print(f"\n지역: {region}")
    for year, birth_count in zip(years_to_predict, pred):
        print(f"{year}년 예측 출생아수: {birth_count:.2f}")
