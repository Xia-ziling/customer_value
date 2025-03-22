# src/data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_preprocess_data(file_path, target_column='Monetary'):
    """
    加载数据、预处理、划分数据集、标准化。

    Args:
        file_path (str): 数据文件路径。
        target_column (str): 目标列名。

    Returns:
        tuple: (X_train_scaled_df, X_test_scaled_df, y_train_df, y_test_df, preprocessor)
               包含标准化后的训练集特征、测试集特征、训练集标签、测试集标签和预处理器。
               所有返回的数据都是 DataFrame 格式。
    """

    # 加载数据
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
    except UnicodeDecodeError:
        print("尝试使用其他编码，例如 UTF-8, latin1 等。")
        return None

    # 数据预处理
    df = df.dropna(subset=['CustomerID'])
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    ref_date = df['InvoiceDate'].max() + pd.DateOffset(1)
    df['Recency'] = (ref_date - df['InvoiceDate']).dt.days
    df['Frequency'] = df.groupby('CustomerID')['InvoiceNo'].transform('nunique')
    df['Monetary'] = df.groupby('CustomerID')['UnitPrice'].transform('sum')

    # 特征选择和异常值处理
    features = ['Recency', 'Frequency', 'Monetary']
    df_selected = df[features]
    df_selected = df_selected[(df_selected['Monetary'] > 0) & (df_selected['Monetary'] < 1000000)]

    # 划分数据集（如果提供了目标列）
    if target_column in df_selected.columns:
        X = df_selected.drop(columns=[target_column])
        y = df_selected[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:  #如果没有提供目标列,则不划分
        return df_selected, None, None, None, None

    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 将标准化后的数据转换回 DataFrame，并保留列名
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    y_train_df = pd.DataFrame(y_train, columns=[target_column])
    y_test_df = pd.DataFrame(y_test, columns=[target_column])

    # 保存预处理器
    joblib.dump(scaler, '../models/preprocessor.pkl')

    return X_train_scaled_df, X_test_scaled_df, y_train_df, y_test_df, scaler