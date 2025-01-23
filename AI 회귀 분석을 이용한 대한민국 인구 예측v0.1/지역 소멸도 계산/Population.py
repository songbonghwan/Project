import pandas as pd

# 데이터 불러오기
population_df = pd.read_csv('csv/지역별 인구수_full.csv')
birth_df = pd.read_csv('Sarimax_csv/예측된_신생아수_2024_2075.csv')
death_df = pd.read_csv('Sarimax_csv/예측된_사망자수_2024_2075.csv')

# 필요한 열 추출
population_df = population_df[['AREA', '2023']]  # 지역과 2023년 인구 수만 추출
birth_df = birth_df[['City'] + [str(year) for year in range(2024, 2076)]]  # 2024년부터 2075년까지 신생아 수
death_df = death_df[['City'] + [str(year) for year in range(2024, 2076)]]  # 2024년부터 2075년까지 사망자 수

# 숫자형으로 변환 (문자열이 포함되어 있을 경우 이를 숫자형으로 변환)
population_df['2023'] = pd.to_numeric(population_df['2023'], errors='coerce')
birth_df.iloc[:, 1:] = birth_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
death_df.iloc[:, 1:] = death_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

# 데이터 결합 (지역 이름 기준으로 병합)
merged_df = population_df.merge(birth_df, left_on='AREA', right_on='City', how='left')
merged_df = merged_df.merge(death_df, left_on='AREA', right_on='City', how='left')

# 2024년 인구수 계산 (2023년 인구수 + 신생아 수 - 사망자 수)
merged_df['2024'] = round(merged_df['2023'] + 
                          merged_df['2024_x'] -  # 신생아 수
                          merged_df['2024_y'])    # 사망자 수

# 2025년부터 2075년까지 인구수 계산 (이주자 수 제외)
for year in range(2025, 2076):
    prev_year = str(year - 1)
    # 이전 연도의 인구수를 기준으로 계산하고, 계산 후 반올림
    merged_df[str(year)] = round(merged_df[prev_year] + 
                                  merged_df[str(year) + '_x'] -  # 신생아 수
                                  merged_df[str(year) + '_y'])    # 사망자 수

# 필요한 열만 추출 (지역명과 각 연도의 인구수만 저장)
result_df = merged_df[['AREA'] + [str(year) for year in range(2024, 2076)]]

# 음수값을 0으로 처리 (저장 전에만 처리)
result_df = result_df.applymap(lambda x: max(x, 0) if isinstance(x, (int, float)) else x)

# 결과 확인
print(result_df.head())

result_df.to_csv('지역별_2024_2075_예상인구.csv', index=False)