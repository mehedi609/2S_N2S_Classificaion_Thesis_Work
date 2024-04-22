from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, classification_report


class ClassificationMatrix:
    def __init__(self, y_true, y_pred, report_name):
        self.y_true = y_true
        self.y_pred = y_pred
        self.report_name = report_name

    def evaluate(self):
        # Calculate and display the confusion matrix
        conf_matrix = confusion_matrix(self.y_true, self.y_pred)
        print(f"Confusion Matrix: ${self.report_name}")
        print(conf_matrix)

        # Calculate and display the accuracy
        accuracy = accuracy_score(self.y_true, self.y_pred)
        print(f"\nAccuracy (ACC): {accuracy:.2f}")

        # Calculate and display the Matthews Correlation Coefficient (MCC)
        mcc = matthews_corrcoef(self.y_true, self.y_pred)
        print(f"Matthews Correlation Coefficient (MCC): {mcc:.2f}")

        # Generate classification report
        report = classification_report(self.y_true, self.y_pred, zero_division=0)
        print("\nClassification Report:")
        print(report)
