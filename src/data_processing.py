# src/data_processing.py
from data_loader import load_and_preprocess_data
import pandas as pd

if __name__ == '__main__':
    # 使用示例：
    file_path = "../data/ecommerce-data.csv"
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(file_path) # 不需要预处理器，用_占位

    if X_train is not None:
        print("数据处理完成。")
        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)

        # 保存处理后的数据到 Excel 文件 (可选)
        with pd.ExcelWriter('../data/processed_data.xlsx', engine='openpyxl') as writer:
            X_train.to_excel(writer, sheet_name='X_train', index=False)
            y_train.to_excel(writer, sheet_name='y_train', index=False)
            X_test.to_excel(writer, sheet_name='X_test', index=False)
            y_test.to_excel(writer, sheet_name='y_test', index=False)
        print("数据保存完毕")
    else:
        print("数据处理失败。")