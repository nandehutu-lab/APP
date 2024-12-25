import streamlit as st
from joblib import load
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的模型
model_path = "lightgbm_model.joblib"  
loaded_model = load(model_path)


# 创建 SHAP 解释器
explainer = shap.TreeExplainer(loaded_model)


# 应用标题
st.title("LightGBM机器学习模型+Hysplit后向轨迹模型+PMF源解析模型+SHAP可解释耦合的在线PM2.5浓度预测及归因应用")

# 描述
st.markdown("""
这是一个基于 Streamlit 的 Web 应用，用户可以通过输入特征值来调用 LightGBM 模型对PM2.5浓度进行预测并进行归因解释。
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
st.header("请对应输入特征数据")
feature_inputs = {}

for feature_name in feature_names:
    feature_inputs[feature_name] = st.number_input(f"{feature_name}:", value=0.0)

# 数据转化为 DataFrame
input_data = pd.DataFrame([feature_inputs.values()], columns=feature_inputs.keys())

# 显示输入数据
st.subheader("输入数据")
st.write(input_data)

# 预测逻辑
if st.button("PM2.5浓度预测"):
    prediction = loaded_model.predict(input_data)
    st.subheader("浓度预测结果")
    st.write(f"浓度预测值: {prediction[0]:.2f}")

    # 生成 SHAP 解释
    shap_values = explainer.shap_values(input_data)
  
    # 转换为 SHAP Explanation 对象
    explanation = shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=input_data.values[0], feature_names=feature_names)
  
    # 显示 SHAP 水瀑图
    st.subheader("运用SHAP算法进行预测解释")
    fig, ax = plt.subplots(figsize=(10, 6))
  
    shap.waterfall_plot(explanation, max_display=25, show=False)
    st.pyplot(fig)  # 使用 Streamlit 展示 matplotlib 图形
