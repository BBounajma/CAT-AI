import pickle
import matplotlib.pyplot as plt
from Models.model_XG_boost import XG_base_model

train_data_path = r"Data\processed_data2.pkl"

with open(train_data_path, 'rb') as f:
    X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(f)

model = XG_base_model

eval_set = [(X_train, y_train),(X_test, y_test)]

model.fit(
    X_train,
    y_train,
    eval_set=eval_set,
    verbose=False
)
model.save_model(r"Models\XG_boost_model.json")

evals = model.evals_result()

train_loss = evals['validation_0']['mlogloss']
val_loss   = evals['validation_1']['mlogloss']
print(f"Final Training Loss: {train_loss[-1]}")
print(f"Final Validation Loss: {val_loss[-1]}")


train_accuracy = sum(model.predict(X_train) == y_train) / len(y_train)
test_accuracy = sum(model.predict(X_test) == y_test) / len(y_test)

print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")

plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Iterations')
plt.ylabel('Log Loss')
plt.title('XGBoost Training and Validation Loss')
plt.legend()
plt.show()
