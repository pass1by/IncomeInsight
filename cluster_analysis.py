import pandas as pd
import numpy as np
import os

def analyze_kmeans_clusters_save_csv(X_pca, X_original, y_train, kmeans_model, output_path='report/cluster_analysis_report.csv'):
    """
    X_pca: PCA降维后的数据（只有PC1, PC2）
    X_original: 原始特征数据（有hours-per-week, capital-gain等）
    y_train: 收入标签
    kmeans_model: 训练好的KMeans模型
    """
    if isinstance(X_pca, np.ndarray):
        X_pca = pd.DataFrame(X_pca)

    X_pca = X_pca[['PC1', 'PC2']]

    # 用PCA数据预测聚类类别
    cluster_labels = kmeans_model.predict(X_pca)

    # 用原始数据做特征分析
    df = X_original.copy()
    df['income'] = y_train
    df['cluster'] = cluster_labels

    feature_names = X_original.columns.tolist()

    records = []

    print("\n===== 聚类类别分析报告 =====")
    for cluster_id in sorted(df['cluster'].unique()):
        sub_df = df[df['cluster'] == cluster_id]
        count = len(sub_df)
        avg_income = sub_df['income'].mean()

        print(f"\n 类别 {cluster_id} 分析：")
        print(f"  样本数: {count}")
        print(f"  高收入比例 (>50K): {avg_income:.2%}")
        print("  特征均值：")

        record = {
            '类别编号': cluster_id,
            '样本数': count,
            '高收入比例(%)': round(avg_income * 100, 2),
            'PC1均值': round(X_pca.iloc[sub_df.index]['PC1'].mean(), 2),
            'PC2均值': round(X_pca.iloc[sub_df.index]['PC2'].mean(), 2),
            'hours-per-week均值': round(sub_df['hours-per-week'].mean(), 2),
            'capital-gain均值': round(sub_df['capital-gain'].mean(), 2),
            'capital-loss均值': round(sub_df['capital-loss'].mean(), 2),
            'age均值': round(sub_df['age'].mean(), 2)
        }

        for feature in feature_names:
            mean_value = sub_df[feature].mean()
            record[f'{feature}均值'] = round(mean_value, 2)
            print(f"    {feature}: {mean_value:.2f}")

        records.append(record)

    result_df = pd.DataFrame(records)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n 聚类分析报告已保存到 {output_path}")
