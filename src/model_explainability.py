# src/model_explainability.py
import shap
import lime
import lime.lime_tabular
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_and_preprocess_data

def explain_model_shap(model, X_train):
    """使用 SHAP 进行全局解释"""
    print("开始 SHAP 全局解释...")  # 添加调试输出

    # 使用 TreeExplainer (适用于树模型)
    if hasattr(model, 'estimators_'):
        # 尝试解释Stacking模型的基学习器
        print("  使用 TreeExplainer...")  # 添加调试输出
        explainer = shap.TreeExplainer(model.estimators_[0])
        print("  开始计算 SHAP 值...")  # 添加调试输出
        shap_values = explainer.shap_values(X_train)
        print("  SHAP 值计算完成。")  # 添加调试输出
    else:
        # 如果不是基于树的模型,使用KernelExplainer(计算较慢)
        print("  使用 KernelExplainer...")  # 添加调试输出
        explainer = shap.KernelExplainer(model.predict, X_train)
        print("  开始计算 SHAP 值...")  # 添加调试输出
        shap_values = explainer.shap_values(X_train)
        print("  SHAP 值计算完成。")  # 添加调试输出

    # 生成 SHAP summary plot
    print("  正在生成 SHAP summary plot...")  # 添加调试输出
    shap.summary_plot(shap_values, X_train, show=False)
    plt.savefig('../models/shap_summary.png')
    plt.close()
    print("  SHAP summary plot 已保存。")  # 添加调试输出

    print("SHAP 全局解释完成。")  # 添加调试输出


def explain_model_lime(model, X_train, X_test, instance_index, feature_names):
    """使用 LIME 进行局部解释"""
    print("开始 LIME 局部解释...")  # 添加调试输出

    # 创建 LIME explainer
    print("  创建 LIME explainer...")  # 添加调试输出
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        mode='regression',
        discretize_continuous=True
    )

    # 解释单个实例
    print(f"  解释实例 {instance_index}...")  # 添加调试输出
    instance = X_test.iloc[instance_index]  # LIME 需要 NumPy 数组

    # 定义一个临时的预测函数
    def predict_fn_temp(x):
        df = pd.DataFrame(x, columns=feature_names)  # 将输入的numpy数组转换为DataFrame
        return model.predict(df)

    explanation = explainer.explain_instance(
        data_row=instance.values,  # 这里需要转换为numpy数组
        predict_fn=predict_fn_temp,  # 使用临时的预测函数
        num_features=len(feature_names)
    )
    print("  实例解释完成。")  # 添加调试输出

    # 生成 LIME 解释图
    print("  正在保存 LIME 解释结果...")  # 添加调试输出
    explanation.save_to_file('../models/lime_explanation.html')
    print("  LIME 解释结果已保存。")  # 添加调试输出

    print("LIME 局部解释完成。")  # 添加调试输出

def model_explain(model, X_train, X_test, instance_index, feature_names):
    """主函数，整合全局和局部的模型解释"""
    # SHAP 全局解释
    explain_model_shap(model, X_train)

    # LIME 局部解释
    explain_model_lime(model, X_train, X_test, instance_index, feature_names)

if __name__ == '__main__':
    # 加载模型
    print("加载模型...")  # 添加调试输出
    model = joblib.load('../models/model.pkl')
    print("模型加载完成。")  # 添加调试输出

    # 加载数据
    print("加载数据...")  # 添加调试输出
    file_path = "../data/ecommerce-data.csv"
    X_train, X_test, _, _, _ = load_and_preprocess_data(file_path)
    print("数据加载完成。")  # 添加调试输出

    # 采样数据
    X_train = X_train[:300]
    X_test = X_test[:300]  # 保持X_test和X_train的行数一致

    # 解释第 1 个测试实例
    instance_index = 1
    feature_names = X_train.columns.tolist()

    # 进行模型解释
    print("开始模型解释...")  # 添加调试输出
    model_explain(model, X_train, X_test, instance_index, feature_names)
    print("模型解释完成。")  # 添加调试输出