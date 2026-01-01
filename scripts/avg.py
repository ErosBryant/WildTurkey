import pandas as pd

# 파일 경로 지정
file_path = '/mnt/lac-sec/20241020-204935/summary_results/summary_results_SST_16Entry_file_error=2_64000000.csv'

# CSV 파일 읽기
df = pd.read_csv(file_path)

# 열 이름에 앞뒤 공백이 있는 경우를 처리하기 위해 공백 제거
df.columns = df.columns.str.strip()

# write_micros/op과 read_micros/op의 평균 계산
average_write_micros = df['write_micros/op'].mean()
average_read_micros = df['read_micros/op'].mean()

# 결과 출력
print(f'Average write_micros/op: {average_write_micros}')
print(f'Average read_micros/op: {average_read_micros}')
