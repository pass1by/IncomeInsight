from preprocess import preprocess_data
from visualization import plot_feature_distributions, plot_correlation_heatmap
from dimensionality_reduction import apply_pca, plot_pca_scatter
from clustering import apply_kmeans, plot_kmeans_clusters
from regression import train_regression_model, evaluate_regression_model
from classification import train_classification_model, evaluate_classification_model
from cluster_analysis import analyze_kmeans_clusters_save_csv
import joblib
import os
if not os.path.exists('models'):
    os.makedirs('models')
def main():
    # 1. 数据预处理
    print("开始数据预处理...")
    train_data, test_data = preprocess_data()
    print("数据预处理完成。")

    # 2. 简单输出数据维度和字段
    print(f"训练集维度: {train_data.shape}")
    print(f"测试集维度: {test_data.shape}")
    print(f"特征列表: {list(train_data.columns)}")

    # 3. 特征分布可视化
    print("开始绘制特征分布图...")
    plot_feature_distributions(train_data)
    print("特征分布图绘制完成，保存到 report/feature_distributions/ 目录下。")

    # 4. 特征相关性热力图
    print("开始绘制特征相关性热力图...")
    plot_correlation_heatmap(train_data)
    print("特征相关性热力图绘制完成，保存到 report/ 目录下。")

     # 3. PCA降维与可视化
    print("开始PCA降维...")
    pca_train_data = apply_pca(train_data)
    print("PCA降维完成。")

    print("绘制PCA降维散点图...")
    plot_pca_scatter(pca_train_data)
    print("PCA降维散点图绘制完成，已保存到 report/ 目录。")

    print("全部处理完成")

    # 4. KMeans聚类
    print("开始KMeans聚类...")
    clustered_data = apply_kmeans(pca_train_data)

    print("绘制KMeans聚类结果散点图...")
    plot_kmeans_clusters(clustered_data)

    # 5. kMeans聚类分析
    print("生成聚类类别统计分析报告...")
    analyze_kmeans_clusters_save_csv(
        X_pca=pca_train_data,  # 传降维后的数据（用于聚类predict）
        X_original=train_data,  # 传原始数据（用于统计hours-per-week等）
        y_train=train_data['income'],  # 真实标签
        kmeans_model=joblib.load('models/kmeans.pkl')
    )
    # 6. 分类模型
    print("开始训练分类模型...")
    clf, X_test, y_test = train_classification_model(train_data)

    print("开始评估分类模型...")
    evaluate_classification_model(clf, X_test, y_test)
    # 7. 回归模型
    print("开始训练回归模型...")
    reg, X_test_reg, y_test_reg = train_regression_model(train_data)

    print("开始评估回归模型...")
    evaluate_regression_model(reg, X_test_reg, y_test_reg)


if __name__ == "__main__":
    main()
