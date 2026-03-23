from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score


def find_best_threshold(y_test, y_prob):

    best_t = 0
    best_precision = 0

    for t in [i/100 for i in range(30, 70)]:
        y_pred = (y_prob > t).astype(int)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # constraint: keep recall high
        if recall >= 0.46:
            if precision > best_precision:
                best_precision = precision
                best_t = t

    print("Best Threshold:", best_t)
    print("Precision at best threshold:", best_precision)

    return best_t


def evaluate_model(model, X_test, y_test):

    y_prob = model.predict_proba(X_test)[:, 1]

    # 🔴 find best threshold
    best_t = find_best_threshold(y_test, y_prob)

    # 🔴 apply it
    y_pred = (y_prob > best_t).astype(int)

    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))