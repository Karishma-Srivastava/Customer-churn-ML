
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):

    # 🔴 Handle imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # 🔴 Model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_res, y_res)

    return model