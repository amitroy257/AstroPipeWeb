from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate(y_true, y_pred, class_names):
    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {"accuracy": acc, "report": report, "confusion_matrix": cm}
