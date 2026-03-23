import pandas as pd
def preprocess_data(df):
    df = df.copy()

    # Drop customerID
    df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    return df