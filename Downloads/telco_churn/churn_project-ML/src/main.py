from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.models.train import train_model
from src.models.evaluate import evaluate_model

def main():
    df = load_data("data/raw/telco.csv")
    df = preprocess_data(df)

    model, X_test, y_test = train_model(df)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()