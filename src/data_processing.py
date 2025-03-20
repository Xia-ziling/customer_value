# src/data_processing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib



def load_data(file_path):
    """加载数据"""
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')  # 尝试不同的编码
        return df
    except UnicodeDecodeError:
        print("尝试使用其他编码，例如 UTF-8, latin1 等。")
        return None


def preprocess_data(df):
    """数据预处理"""
    # 1. 缺失值处理 (根据实际情况填充或删除)
    df = df.dropna(subset=['CustomerID'])  # 示例：删除CustomerID缺失的行

    # 2. 特征工程 (示例：创建RFM特征)
    # 假设 InvoiceDate 是日期时间格式，如果不是，先转换
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # 计算最近日期
    ref_date = df['InvoiceDate'].max() + pd.DateOffset(1)

    # 计算 Recency, Frequency, Monetary
    df['Recency'] = (ref_date - df['InvoiceDate']).dt.days
    df['Frequency'] = df.groupby('CustomerID')['InvoiceNo'].transform('nunique')
    df['Monetary'] = df.groupby('CustomerID')['UnitPrice'].transform('sum')

    # 3. 特征选择/转换 (示例：只保留数值特征)
    features = ['Recency', 'Frequency', 'Monetary']
    df_selected = df[features]

    # 4. 异常值处理（如果需要，根据业务理解处理离群点）
    df_selected = df_selected[(df_selected['Monetary'] > 0) & (df_selected['Monetary'] < 1000000)]  # 去除销售额为负数或者异常大的客户

    return df_selected


def scale_data(X_train, X_test):
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 保存scaler,用于后续部署
    joblib.dump(scaler, '../models/preprocessor.pkl')
    return X_train_scaled, X_test_scaled


def split_data(df, target_column=None):
    """划分数据集, 如果target_column为None，则不划分"""
    if target_column is not None:
        # 确保目标列存在于 DataFrame 中
        if target_column not in df.columns:
            raise ValueError(f"目标列 '{target_column}' 不存在于数据集中")

        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 设置随机种子，确保结果可复现

        # 如果提供了目标列，则划分数据集；否则，不划分
        return X_train, X_test, y_train, y_test
    else:
        # 如果没有提供目标列，则整个数据集作为训练集（或特征矩阵）
        return df, None, None, None  # 整个数据集作为特征矩阵


def data_process(file_path):
    """
    主函数，整合以上的数据处理流程,统一输出
    """
    # 加载和预处理
    df = load_data(file_path)
    if df is None:
        return None, None, None, None
    df_processed = preprocess_data(df)

    # 划分数据集
    X_train, X_test, y_train, y_test = split_data(df_processed, 'Monetary')  # 假设以Monetary为标签
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == '__main__':
    # 使用示例：
    # 假设您的数据文件路径是 ../data/ecommerce-data.csv
    file_path = "../data/ecommerce-data.csv"
    X_train, X_test, y_train, y_test = data_process(file_path)

    if X_train is not None:
        print("数据处理完成。")
        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)
    else:
        print("数据处理失败。")