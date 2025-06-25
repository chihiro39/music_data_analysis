import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 1. 读取 CSV 文件
df = pd.read_csv("nigerian-songs.csv")

# 2. 查看前几行
print(df.head())

# 3. 选择数值特征进行聚类分析
features = ['danceability', 'acousticness', 'loudness', 'speechiness', 'popularity', 'energy']
X = df[features]

# 4. 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 找最优K值（肘部法）
inertia = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

# 绘图查看拐点
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('聚类数量 (k)')
plt.ylabel('Inertia（惯性）')
plt.title('肘部法则选择K值')
plt.show()
# 进行聚类
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 查看各类簇的特征平均值
cluster_profile = df.groupby('cluster')[features].mean()
print(cluster_profile)

# 可视化（以前两个主成分为例）
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=df, palette='Set2')
plt.title('尼日利亚歌曲聚类结果（PCA降维）')
plt.show()