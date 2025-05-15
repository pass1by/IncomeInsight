import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
import os

def prepare_train_data():
    """
    清洗原始adult.data并保存成train_data_processed.csv
    """
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income'
    ]

    if not os.path.exists('../data'):
        print(" 缺少 data/ 文件夹，请检查！")
        return False

    try:
        data = pd.read_csv('../data/adult.data', names=columns, na_values=' ?', skipinitialspace=True)
        print(" adult.data 成功读取！")
    except:
        print(" 无法读取 ../data/adult.data，请确认文件存在！")
        return False

    data = data.dropna()
    print(f" 清洗后数据量：{data.shape}")

    cat_cols = data.select_dtypes(include='object').columns
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
    print("分类变量编码完成！")

    data.to_csv('../data/train_data_processed.csv', index=False)
    print(" train_data_processed.csv 已保存！")
    return True

def prepare_pca_background():
    """
    加载train_data_processed.csv并生成pca_train.npy
    """
    if not os.path.exists('../models'):
        os.makedirs('../models')

    try:
        train_data = pd.read_csv('../data/train_data_processed.csv')
        print(" train_data_processed.csv 成功读取！")
    except:
        print(" 无法读取 train_data_processed.csv，请确认文件存在！")
        return False

    feature_cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                    'marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

    X = train_data[feature_cols]

    try:
        pca = joblib.load('../models/pca.pkl')
        print(" pca.pkl 成功加载！")
    except:
        print(" 无法加载 pca.pkl，请确认文件存在！")
        return False

    X_pca = pca.transform(X)
    np.save('../models/pca_train.npy', X_pca)
    print(" pca_train.npy 已保存！")
    return True

def prepare_all_data():
    """
    总控函数：一键准备train_data_processed.csv和pca_train.npy
    """
    if not os.path.exists('../data/train_data_processed.csv'):
        print(" 缺少 train_data_processed.csv，正在生成...")
        if not prepare_train_data():
            return

    if not os.path.exists('../models/pca_train.npy'):
        print(" 缺少 pca_train.npy，正在生成...")
        if not prepare_pca_background():
            return

    print("所有数据准备完毕！")

if __name__ == "__main__":
    prepare_all_data()
