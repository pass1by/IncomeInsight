import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from preprocess import preprocess_data
import joblib

def apply_pca(data, n_components=2):
    """
    对数据应用PCA降维
    :param data: 原始数据（DataFrame）
    :param n_components: 降维后的维度数（默认降到2维）
    :return: 降维后的DataFrame（含两列 principal_component_1 和 principal_component_2）
    """
    pca = PCA(n_components=n_components)
    # 去掉标签列（一般最后一列是income标签）
    features = data.iloc[:, :-1]
    principal_components = pca.fit_transform(features)
    pca_df = pd.DataFrame(data=principal_components, columns=[f"PC{i+1}" for i in range(n_components)])
    # 也可以把income列加回来方便后续聚类/分类
    pca_df['income'] = data.iloc[:, -1].values
    # 保存PCA模型
    joblib.dump(pca, 'models/pca.pkl')
    print("保存PCA模型成功")
    return pca_df

def plot_pca_scatter(pca_data, output_path='report/pca_scatter.png'):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    """
    绘制PCA降维后的散点图
    :param pca_data: 降维后的DataFrame，必须包含 'PC1', 'PC2', 'income'
    :param output_path: 保存路径
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='income', palette='Set1', data=pca_data)
    plt.title('PCA降维后的二维散点分布图')
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    plt.legend(title='收入类别', loc='best')
    plt.tight_layout()
    # 确保report目录存在
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    print("开始单独测试 dimensionality_reduction.py...")

    # 调用预处理，直接加载数据
    train_data, test_data = preprocess_data()

    # 应用PCA
    print("应用PCA降维...")
    pca_train_data = apply_pca(train_data)

    # 绘制散点图
    print("绘制PCA降维散点图...")
    plot_pca_scatter(pca_train_data)

    print("PCA模块测试完成，降维图已保存到 report/pca_scatter.png")
