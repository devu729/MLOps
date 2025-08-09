import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    print("MSE: ", mse)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "LinearRegressionModel")
    
    print(f"Model logged with {mse} with mse")