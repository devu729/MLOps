from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import joblib

data = load_diabetes()
X, y = data.data, data.target
model = LinearRegression().fit(X, y)

joblib.dump(model, "model.pkl")
print("Saved model as model.pkl - ready for deployment")