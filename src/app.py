# src/app.py
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from data_loader import load_and_preprocess_data  # 导入封装的函数
import os

# 设置页面标题和布局
st.set_page_config(page_title="客户潜在价值评估系统", layout="wide")
st.title("客户潜在价值评估系统")

# 加载模型和预处理器
@st.cache_resource()
def load_model_and_preprocessor():
    model = joblib.load('../models/model.pkl')
    preprocessor = joblib.load('../models/preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_model_and_preprocessor()

# 侧边栏 - 数据上传和参数设置
st.sidebar.header("数据上传")
uploaded_file = st.sidebar.file_uploader("上传 CSV 文件", type=["csv"])

# 检查是否有上传文件
if uploaded_file is not None:
    # 使用封装的函数加载和预处理数据
    df_processed, _, _, _, _ = load_and_preprocess_data(uploaded_file)

    if df_processed is not None:
        # 进行预测
        predictions = model.predict(df_processed)
        df_processed['Predicted_Value'] = predictions

        # 显示预测结果
        st.header("客户价值预测结果")
        st.dataframe(df_processed)

        # 可解释性分析 (选择一个客户进行解释)
        st.header("可解释性分析")
        customer_index = st.selectbox("选择要解释的客户", df_processed.index)

        if st.button("生成解释"):
            # SHAP 值计算
            if hasattr(model, 'estimators_'):
                explainer = shap.TreeExplainer(model.estimators_[0])
                shap_values = explainer.shap_values(df_processed)
            else:
                explainer = shap.KernelExplainer(model.predict, df_processed)
                shap_values = explainer.shap_values(df_processed)

            # SHAP Summary Plot (全局解释)
            st.subheader("SHAP Summary Plot (全局特征重要性)")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, df_processed, show=False)
            st.pyplot(fig)
            plt.clf()

            # SHAP Force Plot (单个客户解释)
            st.subheader(f"SHAP Force Plot (客户 {customer_index})")
            shap.initjs()
            fig = shap.force_plot(explainer.expected_value, shap_values[customer_index], df_processed.iloc[customer_index], matplotlib=True, show=False)
            st.pyplot(fig)
            plt.clf()

            # LIME 解释
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
                st.write("LIME 解释文件不存在，请先生成模型解释。") #通常不会出现该问题，因为已经按下了“生成解释”

    else:
        st.error("数据加载或预处理失败。")
else:
    st.info("请在侧边栏上传数据文件。")