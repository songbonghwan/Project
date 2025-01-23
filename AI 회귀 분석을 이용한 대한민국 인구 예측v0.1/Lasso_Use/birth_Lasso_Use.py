import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score

# 경고 무시 설정
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# 데이터 불러오기
house_price = pd.read_csv('csv/집값(평균)_full.csv', index_col='AREA')
feat_sal = pd.read_csv('csv/feat_sal.csv', index_col='AREA') 
education_expense = pd.read_csv('csv/사교육비 총액_full.csv', index_col='AREA')
birth_rate = pd.read_csv('csv/시군구_출생아수_full.csv', index_col='AREA')
fertile_women = pd.read_csv('csv/가임 여성수_full.csv', index_col='AREA')
marriage_count = pd.read_csv('csv/혼인건수_full.csv', index_col='AREA')

# 데이터 전처리 함수
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

# 지역별로 예측을 진행할 수 있도록 반복문을 추가
regions = house_price.index  # 지역 목록
predictions = {}

# 'ADD_UP' 지역은 제외
regions = regions[regions != 'ADD_UP']

for region in regions:
    region = region.strip()

    print(f"예측을 위한 지역: {region}")

    X = pd.DataFrame({
        'house_price': house_price.loc[region], 
        'wage': feat_sal.loc[region], 
        'education_expense': education_expense.loc[region],
        'fertile_women': fertile_women.loc[region],
        'marriage_count': marriage_count.loc[region]
    })

    y = birth_rate.loc[region]

    # 데이터 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PolynomialFeatures 적용
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

    # Lasso 모델 하이퍼파라미터 튜닝 (GridSearchCV)
    param_grid = {
        'alpha': np.logspace(-4, 1, 6)
    }

    grid_search = GridSearchCV(Lasso(max_iter=10000, random_state=42), param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

    best_alpha = grid_search.best_params_['alpha']
    print(f"Best alpha for Lasso in {region}: {best_alpha}")

    best_model = grid_search.best_estimator_

    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)

    print(f"Train score with best model for {region}: {train_score}")
    print(f"Test score with best model for {region}: {test_score}")

    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"R-squared for {region}: {r2}")

    years_to_predict = np.array([2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])

    X_future = pd.DataFrame({
        'house_price': house_price.loc[region, years_to_predict.astype(str)], 
        'wage': feat_sal.loc[region, years_to_predict.astype(str)],  
        'education_expense': education_expense.loc[region, years_to_predict.astype(str)],
        'fertile_women': fertile_women.loc[region, years_to_predict.astype(str)],
        'marriage_count': marriage_count.loc[region, years_to_predict.astype(str)]
    })

    X_future_scaled = scaler.transform(X_future)
    X_future_poly = poly.transform(X_future_scaled)

    future_predictions = best_model.predict(X_future_poly)
    predictions[region] = future_predictions

# 결과 출력
print("\n2007년부터 2023년까지 각 지역별 예측된 출생아수:")
for region, pred in predictions.items():
    print(f"\n지역: {region}")
    for year, birth_count in zip(years_to_predict, pred): 
        print(f"{year}년 예측 출생아수: {birth_count:.2f}")
