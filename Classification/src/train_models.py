from data_preprocessing import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

def train_model_log_regression(log_reg):
    file_path = "data/churn.csv"

    X_train, X_test, y_train, y_test = preprocessing(file_path)

    return log_reg.fit(X_train, y_train), X_train, X_test, y_train, y_test

def train_model_rand_forest(rand_forest):
    file_path = "data/churn.csv"

    X_train, X_test, y_train, y_test = preprocessing(file_path)

    return rand_forest.fit(X_train, y_train), X_train, X_test, y_train, y_test

def train_model_grad_boost(grad_boost):
    file_path = "data/churn.csv"

    X_train, X_test, y_train, y_test = preprocessing(file_path)

    return grad_boost.fit(X_train, y_train), X_train, X_test, y_train, y_test
