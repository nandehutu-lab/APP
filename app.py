import streamlit as st
from joblib import load
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的模型
model_path = "lightgbm_model.joblib"  # 替换为实际路径
loaded_model = load(model_path)


# 创建 SHAP 解释器
explainer = shap.TreeExplainer(loaded_model)


# 应用标题
st.title("多模型耦合的PM2.5归因预测应用")

# 描述
st.markdown("""
这是一个基于 Streamlit 的 Web 应用，用户可以通过输入特征值来调用 LightGBM 模型对PM2.5归因预测。
""")

# 提供模型的特征名称
feature_names = [  "Ox(NO2+O3)"
                 , "WD(Wind direction)"
                 , "WS(Wind speed)"
                 , "T(Temperature)"
                 , "RH(Relative humidity)"
                 , "P(Pressure)"
                 , "AOD(Aerosol optical depth)"
                 , "BLH(Boundary layer height)"
                 , "SSR(Surface net solar radiation)"
                 , "TCC(Total cloud cover)"
                 , "CC(Coal combustion)"
                 , "Dust(Wind speed)"
                 , "Industrial(Industrial pollution)"
                 , "VE(Vehicular emission)"
                 , "BB(Biomass burning)"
                 , "SIA(Secondary inorganic aerosol)"
                 , "Cluster"
                 ]

# 获取用户输入
st.header("输入特征数据")
feature_inputs = {}

for feature_name in feature_names:
    feature_inputs[feature_name] = st.number_input(f"{feature_name}:", value=0.0)

# 数据转化为 DataFrame
input_data = pd.DataFrame([feature_inputs.values()], columns=feature_inputs.keys())

# 显示输入数据
st.subheader("输入数据")
st.write(input_data)

# 预测逻辑
if st.button("归因预测"):
    prediction = loaded_model.predict(input_data)
    st.subheader("归因预测结果")
    st.write(f"归因预测值: {prediction[0]:.2f}")

    # 生成 SHAP 解释
    shap_values = explainer.shap_values(input_data)
  
    sample_index = 0  
  
    # 显示 SHAP 水瀑图
    st.subheader("SHAP 解释")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap_values[sample_index], max_display=25, show=False)
    st.pyplot(fig)  # 使用 Streamlit 展示 matplotlib 图形
