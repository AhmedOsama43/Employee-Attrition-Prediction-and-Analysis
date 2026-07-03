from sklearn.metrics import classification_report

def evaluate(model, X_test, y_test, threshold=0.4):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print("Threshold:", threshold)
    print("\nReport:\n", classification_report(y_test, y_pred))