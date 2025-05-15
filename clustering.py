import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from preprocess import preprocess_data
from dimensionality_reduction import apply_pca
import joblib

# 🔥 中文显示配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def apply_kmeans(data, n_clusters=2):
    """
    对PCA降维后的数据进行KMeans聚类
    :param data: 降维后的DataFrame（需要有PC1和PC2列）
    :param n_clusters: 聚类簇数
    :return: 添加了'cluster'列的新DataFrame
    """
    features = data[['PC1', 'PC2']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)
    data_with_clusters = data.copy()
    data_with_clusters['cluster'] = clusters
    joblib.dump(kmeans, 'models/kmeans.pkl')
    print("保存KMeans模型成功")

    return data_with_clusters

def plot_kmeans_clusters(clustered_data, output_path='report/kmeans_clusters.png'):
    """
    绘制KMeans聚类后的散点图
    :param clustered_data: 包含'PC1', 'PC2', 'cluster'列的DataFrame
    :param output_path: 保存路径
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='cluster', palette='Set2', data=clustered_data)
    plt.title('KMeans聚类结果散点图')
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    plt.legend(title='聚类簇', loc='best')
    plt.tight_layout()
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    print("开始单独测试 clustering.py...")

    # 加载数据
    train_data, test_data = preprocess_data()

    # 应用PCA
    print("进行PCA降维...")
    pca_train_data = apply_pca(train_data)

    # 应用KMeans聚类
    print("应用KMeans聚类...")
    clustered_data = apply_kmeans(pca_train_data)

    # 绘制聚类散点图
    print("绘制KMeans聚类结果散点图...")
    plot_kmeans_clusters(clustered_data)

    print("KMeans聚类模块测试完成，图表已保存到 report/ 文件夹！")
