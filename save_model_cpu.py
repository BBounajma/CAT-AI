import torch
import joblib
from deeptab.models import SAINTClassifier

model=joblib.load('saint_model.joblib')
# Save the task_model's state_dict
save_path = 'trained_saint.pth'
torch.save(model.task_model.state_dict(), save_path)
print(f"Task model saved to {save_path}.")