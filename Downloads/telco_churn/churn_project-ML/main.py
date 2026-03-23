from src.data.load_data import load_data
from src.data.process.preprocess import preprocess_data
from sklearn.model_selection import train_test_split
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate_model

def main():

    path = r"C:\Users\USE R\Downloads\telco_churn\churn_project-ML\src\data\raw\customer_churn_files.csv"

    df = load_data(path)

    df = preprocess_data(df)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()