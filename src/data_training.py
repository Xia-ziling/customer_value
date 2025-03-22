# src/model_training.py
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
from data_loader import load_and_preprocess_data  # 导入封装的函数


def train_model(X_train, y_train):
    """构建并训练 Stacking 模型"""
    base_learners = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gbm', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ]
    final_learner = Ridge()
    stacking_model = StackingRegressor(estimators=base_learners, final_estimator=final_learner)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(stacking_model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print("Cross-validation RMSE:", rmse_scores)
    print("Mean RMSE:", rmse_scores.mean())

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
    """主函数，整合模型训练和评估流程"""
    # 训练模型
    stacking_model = train_model(X_train, y_train)
    # 评估模型
    evaluate_model(stacking_model, X_test, y_test)
    # 保存模型
    save_model(stacking_model, '../models/model.pkl')


if __name__ == '__main__':
    # 使用示例 (需要先运行 data_processing.py 得到数据)
    file_path = "../data/ecommerce-data.csv"
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(file_path)

    if X_train is not None:
        # 训练模型 (注意：y_train 和 y_test 现在是 DataFrame)
        model_train_and_eval(X_train, X_test, y_train.values.ravel(), y_test.values.ravel())  # 转换为 NumPy 数组
    else:
        print("没有提供标签，无法训练模型。")