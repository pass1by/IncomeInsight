import os
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import preprocess_data

# 3.2 数据可视化与统计分析
# 单特征分布（直方图、饼图、密度图）
#
# 特征间关系（热力图看特征相关性）
#
# 类别分布（比如income>50K和<=50K的人数对比）
# visualization.py
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_feature_distributions(data, output_dir='report/feature_distributions'):
    """
    绘制每个特征的分布图
    - 数值型特征画直方图 (histplot + KDE)
    - 类别型特征画计数图 (countplot)
    保存到指定文件夹
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for col in data.columns:
        plt.figure(figsize=(8, 6))
        if data[col].nunique() <= 10:
            sns.countplot(x=col, data=data)
            plt.title(f"{col} 分布统计")
            plt.xlabel(col)
            plt.ylabel("数量")
        else:
            sns.histplot(data[col], kde=True)
            plt.title(f"{col} 分布直方图")
            plt.xlabel(col)
            plt.ylabel("频率")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{col}_distribution.png")
        plt.close()

def plot_correlation_heatmap(data, output_path='report/correlation_heatmap.png'):
    """
    绘制特征相关性的热力图
    """
    plt.figure(figsize=(12, 10))
    corr = data.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("特征相关性热力图")
    plt.tight_layout()
    # 确保report目录存在
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    print("开始单独测试 visualization.py...")

    # 调用预处理，直接加载处理好的数据
    train_data, test_data = preprocess_data()

    # 测试画特征分布图
    print("绘制特征分布图中...")
    plot_feature_distributions(train_data)

    # 测试画特征相关性热力图
    print("绘制特征相关性热力图中...")
    plot_correlation_heatmap(train_data)

    print("可视化模块测试完成，所有图表已保存")