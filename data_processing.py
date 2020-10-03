import pandas as pd
from sklearn import preprocessing

# 61 columns with na values
# 15 columns with string values
# 56 columns with numbers

d_path = 'cumulative_2020.09.04_15.26.41.csv'
df = pd.read_csv(d_path, header=0)
print(df.columns.tolist())
# removed kepler_name and koi_comment (unnecessary features) features with all nans (koi_longp, koi_ingress, koi_sage, koi_model_dof, koi_model_chisq)
df = df.drop(['kepler_name', 'koi_comment', 'koi_longp', 'koi_ingress', 'koi_sage', 'koi_model_dof', 'koi_model_chisq',
              'koi_pdisposition', 'koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
              'koi_disp_prov'], axis=1)
df = df.dropna()
df = df[df['koi_disposition'] != 'CANDIDATE']
y = df['koi_disposition']
codes, values = pd.factorize(y)
codes_df = pd.DataFrame(codes)
codes_df.to_csv('y_raw.csv')

print(f'{codes}, {values}')

col_with_str = df.select_dtypes(include='object').columns.tolist()
df_nums = df
df_nums.drop(col_with_str, 1, inplace=True)
col_with_num = df_nums.columns.tolist()
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(df_nums)
df_nums = pd.DataFrame(x_scaled)
df_nums.to_csv('x_raw.csv')
