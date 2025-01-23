import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV 파일에서 데이터 불러오기
df = pd.read_csv("csv/가임 여성수_full.csv")
print(df)
# SEJONG의 2012년부터 2023년까지의 데이터 추출
sejong_data = df[df['AREA'] == 'SEJONG'].iloc[:, 1:].values.flatten()

# 결측값이 있는 부분을 NaN으로 설정 (2007, 2008, 2009, 2010, 2011년)
sejong_data_with_nans = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, *sejong_data[5:]], dtype=float)


# NaN을 선형 보간법으로 채우기
x = np.arange(len(sejong_data_with_nans))
valid_data = sejong_data_with_nans[~np.isnan(sejong_data_with_nans)]
valid_x = x[~np.isnan(sejong_data_with_nans)]

# 선형 보간법 사용
from scipy import interpolate

interp_func = interpolate.interp1d(valid_x, valid_data, kind='linear', fill_value="extrapolate")
filled_data = interp_func(x)

# 결과 출력
print("채운 데이터:", filled_data)

# 시각화
plt.plot(filled_data, label='채운 데이터')
plt.title('SEJONG의 2007~2023년 데이터 (선형 보간법으로 결측값 채운 후)')
plt.xlabel('년도')
plt.ylabel('가임 여성수')
plt.xticks(ticks=np.arange(len(sejong_data_with_nans)), labels=[2007 + i for i in range(len(sejong_data_with_nans))])
plt.legend()
plt.show()