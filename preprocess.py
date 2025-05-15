import pandas as pd
from sklearn.preprocessing import LabelEncoder
# 3.1 数据预处理
# 读取数据（pandas）
#
# 缺失值处理（例如 '?' 替换成 np.nan，再drop或者填充）
#
# 分类变量编码（LabelEncoder或One-Hot Encoding）
#
# 数值变量标准化（StandardScaler）
#
# 特征工程（比如组合特征、离散化特征）

columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]

def load_train_data(filepath):
    """
    读取并预处理 adult.data（训练集）
    """
    data = pd.read_csv(filepath, names=columns, na_values=' ?', skipinitialspace=True)
    data = data.dropna()
    return data

def load_test_data(filepath):
    """
    读取并预处理 adult.test（测试集）
    注意跳过第一行，并去除标签中的点号（.）
    """
    data = pd.read_csv(filepath, names=columns, na_values=' ?', skipinitialspace=True, skiprows=1)
    data = data.dropna()
    # 把 income 字段的 '. ' 去掉
    data['income'] = data['income'].str.replace('.', '', regex=False)
    return data

def encode_categorical_columns(data):
    """
    对类别型字段进行Label Encoding编码
    """
    cat_cols = data.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
    return data

def preprocess_data(train_path='data/adult.data', test_path='data/adult.test'):
    """
    综合处理：读取、清洗、编码数据
    返回：处理后的train_data和test_data
    """
    print("Loading training data...")
    train_data = load_train_data(train_path)
    print("Training data loaded:", train_data.shape)

    print("Loading testing data...")
    test_data = load_test_data(test_path)
    print("Testing data loaded:", test_data.shape)

    print("Encoding categorical features...")
    train_data = encode_categorical_columns(train_data)
    test_data = encode_categorical_columns(test_data)
    print("Encoding completed.")

    return train_data, test_data

if __name__ == "__main__":
    train_data, test_data = preprocess_data()
    print("Preprocessing finished.")
    print("Train sample:")
    print(train_data.head())
    print("Test sample:")
    print(test_data.head())