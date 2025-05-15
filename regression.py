import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from preprocess import preprocess_data

# 🔥 中文显示配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def train_regression_model(train_data):
    """
    训练一个随机森林回归器来预测收入
    :param train_data: 清洗编码后的训练数据
    :return: 训练好的回归模型，测试集特征X_test，测试集标签y_test
    """
    X = train_data.drop('income', axis=1)
    y = train_data['income']

    # 切分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练随机森林回归器
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_train)
    # 保存回归模型
    joblib.dump(reg, 'models/regressor.pkl')
    print("保存回归模型成功")
    return reg, X_test, y_test


def evaluate_regression_model(reg, X_test, y_test, output_dir='report'):
    """
    评估回归模型性能，保存回归指标和回归散点图
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    y_pred = reg.predict(X_test)

    # 计算回归指标
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 保存回归指标
    with open(os.path.join(output_dir, 'regression_report.txt'), 'w', encoding='utf-8') as f:
        f.write(f"均方误差 (MSE): {mse:.4f}\n")
        f.write(f"R²得分 (R2 Score): {r2:.4f}\n")

    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"R²得分 (R2 Score): {r2:.4f}")

    # 绘制真实 vs 预测的散点图
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([0, 1], [0, 1], '--', color='gray')  # 理想情况线
    plt.xlabel('真实收入标签')
    plt.ylabel('预测收入标签')
    plt.title('真实 vs 预测 - 回归散点图')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regression_scatter.png'))
    plt.close()


# 🔥 独立测试入口
if __name__ == "__main__":
    print("开始单独测试 regression.py...")

    # 加载数据
    train_data, test_data = preprocess_data()

    # 训练回归模型
    print("训练回归模型...")
    reg, X_test, y_test = train_regression_model(train_data)

    # 评估回归模型
    print("评估回归模型...")
    evaluate_regression_model(reg, X_test, y_test)

    print("回归模块测试完成，回归报告和散点图已保存到 report/ 文件夹！")
