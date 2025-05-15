import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import preprocess_data

# 🔥 中文显示配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def train_classification_model(train_data):
    """
    训练一个随机森林分类器来预测income
    :param train_data: 清洗编码后的训练数据
    :return: 训练好的模型，测试集特征X_test，测试集标签y_test
    """
    X = train_data.drop('income', axis=1)
    y = train_data['income']

    # 切分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练随机森林
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    # 保存分类模型
    joblib.dump(clf, 'models/classifier.pkl')
    print("保存分类模型成功")
    return clf, X_test, y_test


def evaluate_classification_model(clf, X_test, y_test, output_dir='report'):
    """
    评估分类模型性能，保存分类报告和混淆矩阵图
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    y_pred = clf.predict(X_test)

    # 保存分类报告
    report = classification_report(y_test, y_pred, output_dict=False)
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

    print("分类报告:")
    print(report)

    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('随机森林分类器 - 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_confusion_matrix.png'))
    plt.close()

if __name__ == "__main__":
    print("开始单独测试 classification.py...")

    # 加载数据
    train_data, test_data = preprocess_data()

    # 训练模型
    print("训练分类模型...")
    clf, X_test, y_test = train_classification_model(train_data)

    # 评估模型
    print("评估分类模型...")
    evaluate_classification_model(clf, X_test, y_test)

    print("分类模块测试完成，分类报告和混淆矩阵已保存到 report/ 文件夹！")
