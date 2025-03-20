# src/app.py

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from data_processing import load_data, preprocess_data  # 导入预处理的函数
import os

# 设置页面标题和布局
st.set_page_config(page_title="客户潜在价值评估系统", layout="wide")
st.title("客户潜在价值评估系统")


# 加载模型和预处理器
@st.cache(allow_output_mutation=True)
def load_model_and_preprocessor():
    model = joblib.load('../models/model.pkl')
    preprocessor = joblib.load('../models/preprocessor.pkl')
    return model, preprocessor


model, preprocessor = load_model_and_preprocessor()

# 侧边栏 - 数据上传和参数设置
st.sidebar.header("数据上传")
uploaded_file = st.sidebar.file_uploader("上传 CSV 文件", type=["csv"])

# 检查是否有上传文件，以及模型和预处理器是否加载成功
if uploaded_file is not None and model is not None and preprocessor is not None:
    # 数据加载和预处理
    df = load_data(uploaded_file)

    # 检查df是否成功加载
    if df is not None:
        df_processed = preprocess_data(df)

        # 检查特征名称
        feature_names = ['Recency', 'Frequency', 'Monetary']  # 确保与您的特征列名匹配
        missing_features = [feature for feature in feature_names if feature not in df_processed.columns]

        # 只有当必要特征存在时才进行预测
        if not missing_features:
            # 特征选择和转换
            X = df_processed[feature_names]

            # 使用预处理器进行特征缩放
            X_scaled = preprocessor.transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

            # 进行预测
            predictions = model.predict(X_scaled_df)
            df_processed['Predicted_Value'] = predictions

            # 显示预测结果
            st.header("客户价值预测结果")
            st.dataframe(df_processed)

            # 可解释性分析 (选择一个客户进行解释)
            st.header("可解释性分析")
            customer_index = st.selectbox("选择要解释的客户", df_processed.index)

            if st.button("生成解释"):
                # SHAP 值计算 (根据您的模型类型选择合适的 explainer)
                if hasattr(model, 'estimators_'):
                    explainer = shap.TreeExplainer(model.estimators_[0])
                    shap_values = explainer.shap_values(X_scaled_df)
                else:
                    explainer = shap.KernelExplainer(model.predict, X_scaled_df)
                    shap_values = explainer.shap_values(X_scaled_df)

                # SHAP Summary Plot (全局解释)
                st.subheader("SHAP Summary Plot (全局特征重要性)")
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, X_scaled_df, show=False)
                st.pyplot(fig)
                plt.clf()

                # SHAP Force Plot (单个客户解释)
                st.subheader(f"SHAP Force Plot (客户 {customer_index})")
                shap.initjs()  # 初始化 JavaScript 可视化
                fig = shap.force_plot(explainer.expected_value, shap_values[customer_index],
                                      X_scaled_df.iloc[customer_index], matplotlib=True, show=False)
                st.pyplot(fig)
                plt.clf()

                # 由于Streamlit的限制，这里不能直接显示LIME的HTML，而是保存文件
                st.write("LIME 解释已保存为 HTML 文件，请下载查看：")
                # 这里需要确保'lime_explanation.html'存在
                if os.path.exists('../models/lime_explanation.html'):
                    with open('../models/lime_explanation.html', "rb") as file:
                        btn = st.download_button(
                            label="下载 LIME 解释",
                            data=file,
                            file_name="lime_explanation.html",
                            mime="text/html"
                        )
                else:
                    st.write("LIME 解释文件不存在，请先生成模型解释。")  # 通常不会出现该问题，因为已经按下了“生成解释”

        else:
            st.error(f"数据中缺少以下必需特征: {', '.join(missing_features)}")
    else:
        st.error("数据加载失败，请检查文件格式和编码。")
else:
    st.info("请在侧边栏上传数据文件。")