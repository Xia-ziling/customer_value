# src/model_training.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge  # 可以选择回归或分类
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np  # 导入NumPy


def train_model(X_train, y_train):
    """构建并训练 Stacking 模型"""

    # 定义基模型
    base_learners = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gbm', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ]

    # 定义元模型(回归)
    final_learner = Ridge()  # 或者其他回归模型

    # 构建 Stacking 回归模型
    stacking_model = StackingRegressor(estimators=base_learners, final_estimator=final_learner)

    # 交叉验证
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(stacking_model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print("Cross-validation RMSE:", rmse_scores)
    print("Mean RMSE:", rmse_scores.mean())

    # 训练模型
    stacking_model.fit(X_train, y_train)

    return stacking_model


def evaluate_model(model, X_test, y_test):
    """评估模型"""

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Test RMSE: {rmse}")
    print(f"Test R-squared: {r2}")


def save_model(model, filename):
    """保存模型"""
    joblib.dump(model, filename)


def model_train_and_eval(X_train, X_test, y_train, y_test):
    """
    主函数，整合以上模型训练和评估流程
    """

    # 训练模型
    stacking_model = train_model(X_train, y_train)

    # 评估模型
    evaluate_model(stacking_model, X_test, y_test)

    # 保存模型
    save_model(stacking_model, '../models/model.pkl')


if __name__ == '__main__':
    # 使用示例 (需要先运行 data_processing.py 得到数据)
    from data_processing import data_process

    file_path = "../data/ecommerce-data.csv"
    X_train, X_test, y_train, y_test = data_process(file_path)

    # 如果是分类任务，把'Monetary'换成对应的目标列；如果是回归，则可以预测'Monetary'
    # 确保y_train和y_test不是None
    if y_train is not None and y_test is not None:
        # 训练模型
        model_train_and_eval(X_train, X_test, y_train, y_test)
    else:
        print("没有提供标签，无法训练模型。")