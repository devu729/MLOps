import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

mlflow.set_experiment("RandomForest_Experiments")

data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

n_estimators_list = [10, 50, 100]
max_depth_list = [1,3,5]
for n in n_estimators_list:
    for d in max_depth_list:
        with mlflow.start_run():
            model = RandomForestRegressor(n_estimators=n, max_depth=d, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mse = mean_squared_error(y_test, preds)
            mlflow.log_param("n_estimator", n)
            mlflow.log_param("max_depth", d)
            mlflow.log_metric("mse", mse)

            mlflow.sklearn.log_model(model, "Model")
            print(f"Logged model: n_estimator {n}, max_depth {d}, mse {mse}")