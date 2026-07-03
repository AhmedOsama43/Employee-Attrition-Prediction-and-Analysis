import matplotlib.pyplot as plt
import seaborn as sns

def plot_attrition_distribution(df):
    sns.countplot(x='Attrition', data=df)
    plt.title("Attrition Distribution")
    plt.show()

def plot_income_vs_attrition(df):
    sns.boxplot(x='Attrition', y='MonthlyIncome', data=df)
    plt.title("Monthly Income vs Attrition")
    plt.show()

def plot_job_satisfaction(df):
    sns.countplot(x='JobSatisfaction', hue='Attrition', data=df)
    plt.title("Job Satisfaction vs Attrition")
    plt.show()

def plot_correlation(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap='coolwarm')
    plt.title("Feature Correlation")
    plt.show()   

def plot_overtime(df):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.countplot(x='OverTime', hue='Attrition', data=df)
    plt.title("Overtime Impact on Attrition")
    plt.show()           