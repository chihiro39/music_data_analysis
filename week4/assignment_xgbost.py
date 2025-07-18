import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb

# 设置图像显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('US-pumpkins.csv')

# 价格转换和平均值计算
df['Low Price'] = pd.to_numeric(df['Low Price'], errors='coerce')
df['High Price'] = pd.to_numeric(df['High Price'], errors='coerce')
df['Package Price'] = (df['Low Price'] + df['High Price']) / 2

# 清洗与字段选择
df_clean = df[['Package', 'City Name', 'Package Price', 'Variety', 'Date']].dropna()

# 编码城市与品种
le_city = LabelEncoder()
le_variety = LabelEncoder()
df_clean['City Code'] = le_city.fit_transform(df_clean['City Name'])
df_clean['Variety Code'] = le_variety.fit_transform(df_clean['Variety'])

# 提取日期月份
df_clean['Date'] = pd.to_datetime(df_clean['Date'])
df_clean['Month'] = df_clean['Date'].dt.month

# 准备训练数据
X = df_clean[['City Code', 'Variety Code', 'Month']]
y = df_clean['Package Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### LightGBM 模型训练与评估 ###
lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)

lgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
lgb_r2 = r2_score(y_test, y_pred_lgb)

print(f'LightGBM RMSE: {lgb_rmse:.4f}')
print(f'LightGBM R²: {lgb_r2:.4f}')

### XGBoost 模型训练与评估 ###
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
xgb_r2 = r2_score(y_test, y_pred_xgb)

print(f'XGBoost RMSE: {xgb_rmse:.4f}')
print(f'XGBoost R²: {xgb_r2:.4f}')

### 可视化对比 ###
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred_lgb, color='green')
plt.title('LightGBM 预测 vs 实际')
plt.xlabel('实际价格')
plt.ylabel('预测价格')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test, y=y_pred_xgb, color='blue')
plt.title('XGBoost 预测 vs 实际')
plt.xlabel('实际价格')
plt.ylabel('预测价格')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')

plt.tight_layout()
plt.show()

