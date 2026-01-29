import pickle
import unittest 
from Models.model_XG_boost import base_model
from Trainings.train_XG_boost import train_data_path

model=XBoostRegressor()
model.load_model(r"Models\XG_boost_model.json")
with open(train_data_path, 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)
y_pred=model.predict(X_test)

MSE_test=sum((y_test - y_pred) ** 2) / len(y_test)