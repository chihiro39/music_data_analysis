# music_data_analysis
# 尼日利亚歌曲数据的聚类分析项目报告

##  项目简介

本项目旨在使用 **无监督学习中的 K-Means 聚类算法**，对来自 Spotify 的尼日利亚歌曲数据进行聚类分析，以挖掘隐藏在歌曲特征背后的潜在模式。该数据集包含了关于歌曲的多个音频属性，包括舞蹈性、声学、响度、言语性、流行度和活力等指标。

通过对这些特征的标准化处理和聚类建模，我们希望发现不同类型歌曲之间的风格差异，并进行可视化展示。这一项目展示了如何在缺乏标签信息的情境下，借助机器学习方法理解复杂的多维数据结构。

## 数据来源

数据文件：`nigerian-songs.csv`  
数据来源：学习通下载

主要字段包括：

- `danceability`：舞蹈性（越高越适合跳舞）
- `acousticness`：声学性（越高越接近原声音乐）
- `loudness`：响度（单位 dB）
- `speechiness`：言语性（越高越偏向朗诵、说唱）
- `popularity`：歌曲在 Spotify 平台的受欢迎程度（0-100）
- `energy`：活力（节奏、响度和动态）

##  分析方法

1. **数据预处理**：
   - 选取上述六个数值特征作为聚类输入
   - 使用 `StandardScaler` 进行标准化处理，消除量纲影响

2. **肘部法则选取最优聚类数（K）**：
   - 测试 K 从 1 到 10 的聚类效果，通过 Inertia 曲线寻找拐点

3. **KMeans 聚类建模**：
   - 使用 `sklearn.cluster.KMeans` 进行模型训练与预测
   - 将聚类结果作为新特征加入原数据中

4. **主成分分析（PCA）可视化**：
   - 将高维数据压缩至二维空间，用散点图表示各类簇之间的分布差异

##  结果展示

- 成功将尼日利亚歌曲划分为 4 个主要风格簇
- 各类簇在“流行度”“活力”“响度”等维度上存在明显差异
- 可视化结果清晰呈现各个聚类的相对分布

##  技术栈

- **Python 3**
- **Pandas**（数据处理）
- **Scikit-learn**（标准化 + 聚类分析）
- **Matplotlib & Seaborn**（绘图与可视化）
- **PCA 降维**（用于高维可视化）

## 项目价值

- 提供了一种探索**无标签音乐数据集**的思路
- 展示了机器学习中 KMeans 聚类在实际数据分析中的应用流程
- 对 Spotify 平台音乐风格偏好及观众音乐品味进行了建模与解释

