# Employee-Attrition-Prediction-and-Analysis
📌 Project Overview

This project is a two-stage intelligent HR decision-support system designed to:

1️⃣ Predict employee attrition probability
2️⃣ Generate personalized HR intervention recommendations

Unlike traditional ML projects that stop at prediction, this system is:

✔ Predictive
✔ Prescriptive
✔ Explainable
✔ Business-Oriented
🎯 Business Problem

Employee attrition causes:

Recruitment costs

Training expenses

Productivity loss

Knowledge drain

Companies often react after resignation occurs.

This project answers:

Can we detect employees at risk early and recommend data-driven actions to retain them?

🏗️ System Architecture
Employee Data
      │
      ▼
Model 1: Attrition Prediction
(Logistic / RF / XGBoost)
      │
      ▼
Probability Score (0–100%)
      │
      ▼
SHAP Explanation (Top Risk Factors)
      │
      ▼
Model 2: Intelligent Recommendation Engine
      │
      ▼
HR Intervention Suggestions
      │
      ▼
Streamlit Dashboard

🧠 Machine Learning Pipeline
🔹 Stage 1 — Attrition Prediction Model
Algorithms Used:

Logistic Regression (Baseline)

Random Forest

XGBoost (Final Optimized Model)

Handling Imbalance:

SMOTE (Synthetic Minority Oversampling Technique)

Evaluation Metrics:

Accuracy

Precision

Recall

F1-score

ROC-AUC

Confusion Matrix

Cross-validation

🔹 Stage 2 — Intelligent Recommendation Engine

The second model consumes:

Attrition probability

Feature importance insights (SHAP)

Employee profile data

Approach A – Rule-Based Intelligence

Example Logic:
if attrition_prob > 0.7 and overtime == "Yes":
    recommendation = "Reduce overtime workload"

    Approach B – ML-Based Intervention Classifier

Predicts best HR action category:

Salary Adjustment

Work-Life Balance Program

Promotion Path

Training Plan

📊 Dataset

IBM HR Analytics Employee Attrition Dataset

Source: Kaggle

Target: Attrition (Yes/No)

Key Features:

Age

Monthly Income

Job Role

Department

Years at Company

Overtime

Job Satisfaction

Work-Life Balance

Performance Rating

🧪 Mathematical Foundation
Logistic Regression
𝑃
(
𝑌
=
1
∣
𝑋
)
=
1
1
+
𝑒
−
(
𝛽
0
+
𝛽
1
𝑋
1
+
.
.
.
+
𝛽
𝑛
𝑋
𝑛
)
P(Y=1∣X)=
1+e
−(β
0
	​

+β
1
	​

X
1
	​

+...+β
n
	​

X
n
	​

)
1
	​


Used as baseline classification model.

Random Forest

Ensemble of decision trees using bagging:

𝑃
𝑟
𝑒
𝑑
𝑖
𝑐
𝑡
𝑖
𝑜
𝑛
=
Majority Vote of Trees
Prediction=Majority Vote of Trees
XGBoost Objective Function
𝑂
𝑏
𝑗
=
∑
𝑙
(
𝑦
𝑖
,
𝑦
𝑖
^
)
+
∑
Ω
(
𝑓
𝑘
)
Obj=∑l(y
i
	​

,
y
i
	​

^
	​

)+∑Ω(f
k
	​

)

Where:

𝑙
l = Loss function

Ω
Ω = Regularization term

🔍 Model Explainability (SHAP)

SHAP values allow:

Global feature importance analysis

Individual prediction explanation

Business transparency

Example Insight:

High overtime + low job satisfaction → High attrition risk

🛠️ Tech Stack
Category	Tools
Language	Python
Data Analysis	Pandas, NumPy
Visualization	Matplotlib, Seaborn
ML	Scikit-learn, XGBoost
Imbalance	SMOTE
Explainability	SHAP
IDE	VS Code
Deployment	Streamlit
📂 Project Structure
employee-attrition-project/
│
├── data/
│   └── raw_dataset.csv
│
├── models/
│   ├── attrition_model.pkl
│   └── recommendation_model.pkl
│
├── notebooks/ (optional exploration)
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── recommendation_engine.py
│   └── utils.py
│
├── app.py
├── main.py
├── requirements.txt
└── README.md
🚀 How to Run
1️⃣ Clone Repository
git clone https://github.com/your-username/employee-attrition-project.git
cd employee-attrition-project
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run Application
streamlit run app.py
📈 Expected Results

≥ 80% Prediction Accuracy

Clear identification of top attrition drivers

Intelligent personalized HR recommendations

Fully interactive decision-support dashboard

💼 Business Impact

This system enables HR departments to:

Reduce turnover rate

Minimize recruitment cost

Improve employee engagement

Apply targeted retention strategies

🔮 Future Enhancements

Cloud Deployment (AWS / Azure)

Real-time HR integration

Cost simulation per employee

Deep learning extension

Dashboard analytics for executives

📊 Sample Dashboard (To Add)

📌 Add screenshots here:

Prediction interface

SHAP explanation plot

Confusion Matrix

Recommendation output

⭐ Why This Project is Advanced

✔ Two-stage intelligent system
✔ Explainable AI
✔ Business-ready architecture
✔ Clean modular structure
✔ Deployment-ready
✔ Combines Prediction + Decision Support
