import os
import joblib
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import metrics


def evaluate_model(y_test, y_pred):
    print("\033[32mModel Evaluation\033[0m\n")

    print("\033[33mMean Absolute Error\033[0m")
    print(metrics.mean_absolute_error(y_test, y_pred), "\n")

    print("\033[33mMean Squared Error\033[0m")
    print(metrics.mean_squared_error(y_test, y_pred), "\n")

    print("\033[33mRoot Mean Squared Error\033[0m")
    print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), "\n")

    print("\033[33mR2 Score\033[0m")
    print(metrics.r2_score(y_test, y_pred), "\n")


def save_model(model, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(model, filename)


def main():
    data = load_data('data/EmployeeBurnout_Preprocessed.csv')

    X = data.drop('Burn Rate', axis=1)
    Y = data['Burn Rate']

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=0)

    # Random Forest Regressor
    rfr = RandomForestRegressor(n_estimators=100, random_state=0)
    rfr.fit(x_train, y_train)
    y_pred = rfr.predict(x_test)

    print("\033[34mRandom Forest Regressor\033[0m\n")
    evaluate_model(y_test, y_pred)

    # SVM Regressor

    # Linear Kernel
    svr_linear = SVR(kernel='linear')
    svr_linear.fit(x_train, y_train)
    y_pred = svr_linear.predict(x_test)

    print("\033[34mSupport Vector Machine Regressor (Linear Kernel)\033[0m\n")
    evaluate_model(y_test, y_pred)

    # RBF Kernel
    svr_rbf = SVR(kernel='rbf')
    svr_rbf.fit(x_train, y_train)
    y_pred = svr_rbf.predict(x_test)

    print("\033[34mSupport Vector Machine Regressor (RBF Kernel)\033[0m\n")
    evaluate_model(y_test, y_pred)

    # Poly Kernel
    svr_poly = SVR(kernel='poly')
    svr_poly.fit(x_train, y_train)
    y_pred = svr_poly.predict(x_test)

    print("\033[34mSupport Vector Machine Regressor (Poly Kernel)\033[0m\n")
    evaluate_model(y_test, y_pred)

    # Save Models
    save_model(rfr, 'models/RandomForestRegressor.pkl')
    save_model(svr_linear, 'models/SVR_Linear.pkl')
    save_model(svr_rbf, 'models/SVR_RBF.pkl')
    save_model(svr_poly, 'models/SVR_Poly.pkl')


if __name__ == "__main__":
    main()
