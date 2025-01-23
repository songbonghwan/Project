import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# CSV 파일에서 데이터 불러오기
df = pd.read_csv("csv/이주데이터_전입.csv")

# 'AREA'가 SEJONG인 데이터 추출
sejong_data = df[df['AREA'] == 'SEJONG'].iloc[:, 1:]

# 쉼표 제거 후 숫자 형식으로 변환 (apply로 각 셀을 처리)
sejong_data = sejong_data.applymap(lambda x: str(x).replace(",", "") if isinstance(x, str) else x)
sejong_data = sejong_data.apply(pd.to_numeric, errors='coerce')

# 2012년부터 2023년까지의 유효한 데이터를 추출
valid_data = sejong_data.iloc[:, 5:].dropna(axis=1)

# 2007년부터 2011년까지의 데이터 (NaN이 포함된 부분)
missing_data = sejong_data.iloc[:, :5]

# 2012년부터 2023년까지의 데이터로 선형 회귀 모델 학습
years_valid = np.array(valid_data.columns.astype(int)).reshape(-1, 1)
population_valid = valid_data.iloc[0].values.reshape(-1, 1)

model = LinearRegression()
model.fit(years_valid, population_valid)

# 2007년부터 2011년까지 예측
years_missing = np.array([2007, 2008, 2009, 2010, 2011]).reshape(-1, 1)
predictions = model.predict(years_missing)

# 예측된 값을 2007-2011에 채워넣기
sejong_data.iloc[0, :5] = predictions.flatten()

# 예측 결과 출력
print("예측된 2007년부터 2011년까지의 인구수:", predictions.flatten())

# 전체 데이터 시각화 (2012년부터 2023년까지 + 예측된 2007년부터 2011년까지)
plt.plot(np.concatenate([years_missing.flatten(), years_valid.flatten()]), 
         np.concatenate([predictions.flatten(), population_valid.flatten()]), label='선형 예측')
plt.scatter(years_valid.flatten(), population_valid.flatten(), color='blue', label='2012-2023 실제 데이터')
plt.scatter(years_missing.flatten(), predictions.flatten(), color='red', label='예측된 2007-2011 데이터')

plt.title('SEJONG 인구수 예측 (2007-2023)')
plt.xlabel('년도')
plt.ylabel('인구수')
plt.legend()
plt.show()
