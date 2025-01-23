import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

# 경고 무시 (모델 수렴 경고 등)
warnings.simplefilter('ignore')

# 임금 데이터 불러오기
feat_sal = pd.read_csv('csv/feat_sal.csv', index_col='AREA')

# 원하는 도시 순서 지정
city_order = [
    'SEOUL', 'BUSAN', 'DAEGU', 'INCHEON', 'GWANGJU', 'DAEJEON', 'ULSAN', 'SEJONG',
    'GYEONGGI', 'GANGWON', 'CHUNGBUK', 'CHUNGNAM', 'JEONBUK', 'JEONNAM',
    'GYEONGBUK', 'GYEONGNAM', 'JEJU'
]

# 도시별로 시계열 예측 수행 (ADD_UP을 제외)
cities = feat_sal.index.unique()
cities = [city for city in cities if city != 'ADD_UP']  # 'ADD_UP'을 제외한 도시들만 선택

# 예측할 연도 범위 (2024년부터 2075년까지)
future_years = list(range(2024, 2076))

# 결과 저장용 딕셔너리
predictions = {}
forecast_results = []  # 예측 결과를 저장할 리스트

for city in city_order:  # 지정된 도시 순서대로 반복
    if city in cities:  # 해당 도시가 데이터에 있을 경우에만 진행
        # 각 도시의 데이터 선택
        city_data = feat_sal.loc[city]  # drop 필요 없음
        
        # 데이터를 DataFrame으로 변환 (이제 city_data는 Series에서 DataFrame으로 변환)
        city_data = city_data.to_frame().T  # Series -> DataFrame 변환 후 전치
        city_data.columns = [str(year) for year in city_data.columns]  # 열 이름을 연도로 설정
        
        # 연도 인덱스를 datetime 형식으로 변환
        city_data.columns = pd.to_datetime(city_data.columns, format='%Y')

        # 데이터에서 쉼표를 제거하고 숫자로 변환 (e.g., 7,577,110.50 -> 7577110.50)
        city_data = city_data.replace({',': ''}, regex=True).astype(float)

        # NaN 값 처리 (NaN을 이전 값으로 대체)
        city_data = city_data.fillna(method='ffill')

        # 로그 변환 (임금의 범위 차이를 줄이기 위함)
        city_data = np.log(city_data)

        # 최적의 SARIMAX 모델 파라미터를 자동으로 찾기 (auto_arima 사용)
        model_auto = auto_arima(city_data.iloc[0], seasonal=False, stepwise=True, trace=True, suppress_warnings=True)

        # 최적의 파라미터를 사용하여 SARIMAX 모델 학습
        best_order = model_auto.order  # (p, d, q)
        
        # SARIMAX 모델 정의 (추세와 계절성 고려 안 함)
        model = SARIMAX(city_data.iloc[0], 
                        order=best_order,  # 최적 파라미터로 설정
                        trend='c',  # 선형 추세 반영
                        enforce_stationarity=True,     # 모델의 안정성 강제
                        enforce_invertibility=True)    # 모델의 역변환 가능성 강제
        model_fit = model.fit(disp=False)

        # 예측 수행 (2024-2075년)
        forecast = model_fit.get_forecast(steps=52)
        forecast_values = forecast.predicted_mean

        # 로그 값을 원래 값으로 복원
        forecast_values_original = np.exp(forecast_values)

        # 예측값 변화율 제한 (3% 이상 증가하지 않도록 제한)
        for i in range(1, len(forecast_values_original)):
            max_increase = forecast_values_original[i-1] * 1.03  # 3% 이상 증가하지 않도록 제한
            if forecast_values_original[i] > max_increase:
                forecast_values_original[i] = max_increase

        # 예측값 감소폭 제한 (급격한 감소를 방지하기 위해)
        for i in range(1, len(forecast_values_original)):
            max_decrease = forecast_values_original[i-1] * 0.97  # 3% 이하로 감소하도록 제한
            if forecast_values_original[i] < max_decrease:
                forecast_values_original[i] = max_decrease

        # 예측 결과를 평활화 (급격한 변동을 줄이기 위해 이동평균 적용)
        smoothed_forecast_values = pd.Series(forecast_values_original).rolling(window=6, min_periods=1).mean()  # 평활화 윈도우 6으로 확대

        # 예측 결과 저장
        predictions[city] = smoothed_forecast_values

        # 예측 결과를 CSV용으로 저장 (future_years와 함께)
        city_forecast = pd.DataFrame({
            'City': [city] * len(future_years),
            'Year': future_years,
            'Forecasted Salary': smoothed_forecast_values
        })
        forecast_results.append(city_forecast)

# 예측된 데이터 표로 출력
forecast_results_df = pd.concat(forecast_results, ignore_index=True)

# 데이터를 연도별로 열로 배치하기 위해 피벗
forecast_pivot = forecast_results_df.pivot(index='City', columns='Year', values='Forecasted Salary')

# 도시 순서를 지정된 순서대로 정렬
forecast_pivot = forecast_pivot.reindex(city_order)

# 예측된 데이터를 연도별로 열로 변환된 표로 출력
print(forecast_pivot)

# CSV 파일로 저장
forecast_pivot.to_csv('예측된_임금_총액_2024_2075.csv')

# 시각화
plt.figure(figsize=(12, 8))

# 도시별 예측 결과 그래프 그리기
for city in predictions:
    plt.plot(future_years, predictions[city], label=city, marker='o')

plt.title('2024년부터 2075년까지의 도시별 임금 예측')
plt.xlabel('Year')
plt.ylabel('Salary')
plt.legend(title='Cities')
plt.grid(True)
plt.xticks(future_years, rotation=45)  # x축 년도 표시 간격 조정
plt.tight_layout()

# 그래프 출력
plt.show()
