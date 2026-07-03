import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):
    df = df.copy()

    # Delete Useless Coloumns
    df.drop(['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount'], axis=1, inplace=True)

    # Convert Target to 1 & 0
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    # Encoding
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    return df