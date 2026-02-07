from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def preprocessing(file_path):
    BASE_DIR = Path(__file__).resolve().parent.parent
    csv_path = BASE_DIR / file_path
    df = pd.read_csv(csv_path)

    #missing values and #data types
    col_str = df["TotalCharges"].astype(str)
    numeric_converted = pd.to_numeric(col_str.str.strip(), errors="coerce")
    df["TotalCharges"] = numeric_converted

    #drop id column
    df = df.drop(columns=["customerID"])

    #delete lines due to TotalCharges attribute fixing 
    df = df.dropna()

    #encoding columns
    df["gender"] = df["gender"].map({"Female": 0, "Male": 1})
    df["Partner"] = df["Partner"].map({"No": 0, "Yes": 1})
    df["Dependents"] = df["Dependents"].map({"No": 0, "Yes": 1})
    df["PhoneService"] = df["PhoneService"].map({"No": 0, "Yes": 1})

    df["MultipleLines"] = df["MultipleLines"].str.strip().str.title()
    df["MultipleLines"] = df["MultipleLines"].map({"No": 0, "Yes": 1, "No Phone Service": 0})

    df["InternetService"] = df["InternetService"].str.strip().str.title()
    df = pd.get_dummies(df, columns=["InternetService"], drop_first=True, dtype=int)

    df["OnlineSecurity"] = df["OnlineSecurity"].str.strip().str.title()
    df["OnlineSecurity"] = df["OnlineSecurity"].map({"No": 0, "Yes": 1, "No Internet Service": 0})

    df["OnlineBackup"] = df["OnlineBackup"].str.strip().str.title()
    df["OnlineBackup"] = df["OnlineBackup"].map({"No": 0, "Yes": 1, "No Internet Service": 0})

    df["DeviceProtection"] = df["DeviceProtection"].str.strip().str.title()
    df["DeviceProtection"] = df["DeviceProtection"].map({"No": 0, "Yes": 1, "No Internet Service": 0})

    df["TechSupport"] = df["TechSupport"].str.strip().str.title()
    df["TechSupport"] = df["TechSupport"].map({"No": 0, "Yes": 1, "No Internet Service": 0})

    df["StreamingTV"] = df["StreamingTV"].str.strip().str.title()
    df["StreamingTV"] = df["StreamingTV"].map({"No": 0, "Yes": 1, "No Internet Service": 0})

    df["StreamingMovies"] = df["StreamingMovies"].str.strip().str.title()
    df["StreamingMovies"] = df["StreamingMovies"].map({"No": 0, "Yes": 1, "No Internet Service": 0})


    df["Contract"] = df["Contract"].str.strip().str.title()
    df = pd.get_dummies(df, columns=["Contract"], drop_first=True, dtype=int)

    df["PaperlessBilling"] = df["PaperlessBilling"].map({"No": 0, "Yes": 1})

    df["PaymentMethod"] = df["PaymentMethod"].str.strip().str.title()
    df = pd.get_dummies(df, columns=["PaymentMethod"], drop_first=True, dtype=int)

    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    #df.info()

    #check class inbalance
    count_yes = 0
    count_no = 0

    for churn in df["Churn"]:
        if churn == 1:
            count_yes+=1
        else:
            count_no+=1

    print(count_yes*100 / (count_no+count_yes))

    # Spliting train and test
    y = df["Churn"]  

    # Features (drop the target column)
    X = df.drop(columns=["Churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Only scale numeric columns
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    return X_train, X_test, y_train, y_test
