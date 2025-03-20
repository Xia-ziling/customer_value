# src/model_explainability.py
import shap
import lime
import lime.lime_tabular
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def explain_model_shap(model, X_train):
    """使用 SHAP 进行全局解释"""

    # 使用 TreeExplainer (适用于树模型)
    if hasattr(model, 'estimators_'):  # 检查是否有基学习器
        # 尝试解释Stacking模型的基学习器
        explainer = shap.TreeExplainer(model.estimators_[0])
        shap_values = explainer.shap_values(X_train)
    else:
        # 如果不是基于树的模型,使用KernelExplainer(计算较慢)
        explainer = shap.KernelExplainer(model.predict, X_train)
        shap_values = explainer.shap_values(X_train)

    # 生成 SHAP summary plot
    shap.summary_plot(shap_values, X_train, show=False)
    plt.savefig('../models/shap_summary.png')
    plt.close()  # 关闭图像


def explain_model_lime(model, X_train, X_test, instance_index, feature_names):
    """使用 LIME 进行局部解释"""
    # 创建 LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        mode='regression',  # 或 'classification'
        discretize_continuous=True
    )

    # 解释单个实例
    instance = X_test[instance_index]
    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict,
        num_features=len(feature_names)  # 显示所有特征
    )

    # 生成 LIME 解释图
    explanation.save_to_file('../models/lime_explanation.html')


def model_explain(model, X_train, X_test, instance_index, feature_names):
    """
    主函数，整合全局和局部的模型解释
    """
    # SHAP 全局解释
    explain_model_shap(model, X_train)

    # LIME 局部解释
    explain_model_lime(model, X_train, X_test, instance_index, feature_names)


if __name__ == '__main__':
    # 使用示例 (需要先运行 model_training.py 训练模型)
    model = joblib.load('../models/model.pkl')
    # 加载预处理器
    preprocessor = joblib.load('../models/preprocessor.pkl')

    # 假设X_train_scaled是未缩放的原始数据
    X_train, _, _, _ = pd.read_csv('../data/ecommerce-data.csv', encoding='ISO-8859-1'), None, None, None  # 从csv中加载
    X_train = X_train[['Recency', 'Frequency', 'Monetary']]  # 仅保留特征列
    X_train = X_train.dropna()  # 删除缺失值
    X_train = X_train[(X_train['Monetary'] > 0) & (X_train['Monetary'] < 1000000)]  # 去除销售额为负数或者异常大的客户

    # 对特征进行缩放
    X_train_scaled = preprocessor.transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

    _, X_test, _, _ = pd.read_csv('../data/ecommerce-data.csv', encoding='ISO-8859-1'), None, None, None  # 从csv中加载
    X_test = X_test[['Recency', 'Frequency', 'Monetary']]  # 仅保留特征列
    X_test = X_test.dropna()  # 删除缺失值
    X_test = X_test[(X_test['Monetary'] > 0) & (X_test['Monetary'] < 1000000)]

    # 对特征进行缩放
    X_test_scaled = preprocessor.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # 解释第 1 个测试实例
    instance_index = 1
    feature_names = X_train.columns.tolist()  # 确保特征名称与数据一致

    # 进行模型解释
    model_explain(model, X_train_scaled, X_test_scaled.values, instance_index, feature_names)  # X_test需要转化为numpy array