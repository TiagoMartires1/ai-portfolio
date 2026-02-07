from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from train_models import train_model_log_regression, train_model_rand_forest, train_model_grad_boost
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def evaluate_model(trained_model, X_train, X_test, y_train, y_test):

    y_pred = trained_model.predict(X_test)
    y_train_pred = trained_model.predict(X_train)

    # Cross-Validation
    cv_scores = cross_val_score(
        trained_model,
        X_train,
        y_train,
        cv=5,
        scoring="roc_auc"
    )

    print("\n--- Cross Validation (5-Fold) ---")
    print("CV ROC-AUC scores:", cv_scores)
    print("Mean CV ROC-AUC :", cv_scores.mean())
    print("Std CV ROC-AUC  :", cv_scores.std())

    # metrics
    # Test Confusion matrix Accuracy, precision, recall, F1
    print("\nTest Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Test Accuracy :", accuracy_score(y_test, y_pred))
    print("Test Precision:", precision_score(y_test, y_pred))
    print("Test Recall   :", recall_score(y_test, y_pred))
    print("Test F1 Score :", f1_score(y_test, y_pred))

    # Train Confusion matrix Accuracy, precision, recall, F1
    print("\nTrain Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))
    print("Train Accuracy :", accuracy_score(y_train, y_train_pred))
    print("Train Precision:", precision_score(y_train, y_train_pred))
    print("Train Recall   :", recall_score(y_train, y_train_pred))
    print("Train F1 Score :", f1_score(y_train, y_train_pred))

    # ROC-AUC# Predict probabilities
    y_prob = trained_model.predict_proba(X_test)[:, 1]
    y_train_prob = trained_model.predict_proba(X_train)[:, 1]

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_prob)

    # AUC score
    auc_score = roc_auc_score(y_test, y_prob)
    auc_score_train = roc_auc_score(y_train, y_train_prob)
    print("Test ROC-AUC Score:", auc_score)
    print("Train ROC-AUC Score:", auc_score_train)

    # Plot
    plt.figure()
    plt.plot(fpr, tpr, label="Test Model")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(fpr_train, tpr_train, label="Train Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="g")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Train vs Test ROC Curve")
    plt.legend()
    plt.show()


# Create the models
lr_model = LogisticRegression(max_iter=1000)

rf_model = RandomForestClassifier(
    n_estimators=200,     # number of trees
    max_depth=None,      # allow full depth
    random_state=42,
    class_weight="balanced"  # useful for churn imbalance
)

rf_model_tuned = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=10,
    random_state=42,
    class_weight="balanced"
)

gb_model = GradientBoostingClassifier(
    n_estimators=300,        # more trees
    learning_rate=0.05,      # slower learning, less overfit
    max_depth=4,             # shallow trees, prevent overfitting
    min_samples_leaf=15,     # avoid tiny leaves
    subsample=0.8,           # row sampling to reduce variance
    max_features="sqrt",     # feature sampling
    random_state=42
)

#train models
trained_lr, X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_model_log_regression(lr_model)
trained_rf, X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_model_rand_forest(rf_model)
trained_rf_tuned, X_train_rf_tuned, X_test_rf_tuned, y_train_rf_tuned, y_test_rf_tuned = train_model_rand_forest(rf_model_tuned)
trained_gb, X_train_gb, X_test_gb, y_train_gb, y_test_gb = train_model_grad_boost(gb_model)

evaluate_model(trained_lr, X_train_lr, X_test_lr, y_train_lr, y_test_lr)
#evaluate_model(trained_rf, X_train_rf, X_test_rf, y_train_rf, y_test_rf)
#evaluate_model(trained_rf_tuned, X_train_rf_tuned, X_test_rf_tuned, y_train_rf_tuned, y_test_rf_tuned)
#evaluate_model(trained_gb, X_train_gb, X_test_gb, y_train_gb, y_test_gb)
