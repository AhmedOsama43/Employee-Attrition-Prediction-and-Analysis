import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time

from src.data_preprocessing import load_data, preprocess_data
from src.train_model import train_model
from src.recommendation_engine import generate_recommendation

# Page Config
st.set_page_config(page_title="Attrition AI", page_icon="", layout="wide")

# Custom CSS
st.markdown("""
<style>
.big-title {
    font-size:40px;
    font-weight:bold;
    color:#4CAF50;
}
</style>
""", unsafe_allow_html=True)

# Logo + Title
st.image("assets/logo.png", width=100)

st.markdown('<p class="big-title">Employee Attrition AI System</p>', unsafe_allow_html=True)
st.write("Smart HR Decision Support Dashboard")

# Load Data
df = load_data("data/employee_attrition.csv")
df_processed = preprocess_data(df)

model, X_test, y_test = train_model(df_processed)

THRESHOLD = 0.4

# Layout
col1, col2 = st.columns(2)

# Inputs
with col1:
    st.subheader("Employee Info")

    age = st.slider("Age", 18, 60, 30)
    income = st.number_input("Monthly Income", 1000, 20000, 5000)
    overtime = st.selectbox("OverTime", ["Yes", "No"])
    job_sat = st.slider("Job Satisfaction", 1, 4, 3)
    work_life = st.slider("Work Life Balance", 1, 4, 3)

    predict_btn = st.button("Predict")

# Output
with col2:
    st.subheader("Results")

    if predict_btn:

        with st.spinner("Analyzing employee data..."):
            time.sleep(1)

        input_dict = {
            "Age": age,
            "MonthlyIncome": income,
            "OverTime": 1 if overtime == "Yes" else 0,
            "JobSatisfaction": job_sat,
            "WorkLifeBalance": work_life
        }

        input_df = pd.DataFrame([input_dict])

        for col in X_test.columns:
            if col not in input_df.columns:
                input_df[col] = df_processed[col].mean()

        input_df = input_df[X_test.columns]

        prob = model.predict_proba(input_df)[0][1]
        prediction = 1 if prob >= THRESHOLD else 0

        
        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Attrition Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 40], 'color': "green"},
                    {'range': [40, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "red"},
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)


        # Result
        if prediction == 1:
            st.error("High Risk of Leaving")
        else:
            st.success("Employee Likely to Stay")
            st.balloons()

        st.metric("Attrition Probability", f"{prob:.2f}")

        # Recommendations
        st.subheader("Recommendations")

        recs = generate_recommendation(input_df.iloc[0], prob, THRESHOLD)

        if recs:
            for rec in recs:
                st.warning(f"{rec}")
        else:
            st.info("No actions needed")


# Charts Section
st.subheader("Data Insights")

c1, c2 = st.columns(2)

with c1:
    fig1 = px.histogram(df, x="Attrition", title="Attrition Distribution")
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    fig2 = px.box(df, x="Attrition", y="MonthlyIncome", title="Income vs Attrition")
    st.plotly_chart(fig2, use_container_width=True)

# Feature Importance
st.subheader("Top Factors")

importance = model.feature_importances_
features = X_test.columns

feat_imp = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False).head(8)

st.bar_chart(feat_imp.set_index("Feature"))