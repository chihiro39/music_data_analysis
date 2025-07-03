import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 读取数据
df = pd.read_csv('US-pumpkins.csv')

# 转换价格列为数字，并创建平均价格列
df['Low Price'] = pd.to_numeric(df['Low Price'], errors='coerce')
df['High Price'] = pd.to_numeric(df['High Price'], errors='coerce')
df['Package Price'] = (df['Low Price'] + df['High Price']) / 2

# 删除空值并选取关键字段（注意没有 Class Code）
df_clean = df[['Package', 'City Name', 'Package Price', 'Variety', 'Date']].dropna()

# 标签编码
le = LabelEncoder()
df_clean['City Code'] = le.fit_transform(df_clean['City Name'])
df_clean['Variety Code'] = le.fit_transform(df_clean['Variety'])

# Seaborn 箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(x='Variety', y='Package Price', data=df_clean)
plt.title('不同南瓜品种的价格分布（Seaborn）')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Matplotlib 柱状图（城市均价 Top10）
city_avg = df_clean.groupby('City Name')['Package Price'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
city_avg.plot(kind='bar', color='orange')
plt.ylabel('平均价格')
plt.title('各城市南瓜平均价格（Matplotlib）')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

