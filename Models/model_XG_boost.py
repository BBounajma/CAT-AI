from xgboost import XGBClassifier, XGBRegressor
from sklearn.base import clone
import numpy as np

XG_base_model = XGBClassifier(
    n_estimators=130,
    max_depth=6,
    learning_rate=0.1,   
    gamma=0,
    min_child_weight=1,
    objective='multi:softprob',
    num_class=5,
    eval_metric='mlogloss',   
    tree_method='hist',
    random_state=42
)

XG_regressor_model = XGBRegressor(
    n_estimators=130,
    max_depth=6,
    learning_rate=0.1,   
    gamma=0,
    min_child_weight=1,
    objective='reg:squarederror',
    eval_metric='rmse',   
    tree_method='hist',
    random_state=42
)

class xgb_ensemble():

    def fit(self,x_smp,y_smp):
        self.y = [y_smp[:,i].reshape(-1) for i in range(y_smp.shape[1])]
        self.x = x_smp
        self.module_list = [XGBRegressor() for i in range(y_smp.shape[1])]
        self.module_len = len(self.module_list)
        for i in range(self.module_len):
            self.module_list[i].fit(self.x,self.y[i])
            
    def predict(self,new_x):
        y_predict_list  = [self.module_list[i].predict(new_x) for i in range(self.module_len)]
        tmp = y_predict_list[0].reshape(-1,1)
        for i in range(1,self.module_len):
            tmp = np.concatenate((tmp,y_predict_list[i].reshape(-1,1)),1)
        y_predict = tmp
        return y_predict
    

class multiclass_XG_regressor_model:
    def __init__(self, number_of_classes=5):
        self.number_of_classes = number_of_classes
        self.models = {}
        for i in range(1, number_of_classes):
            self.models[i] = clone(XG_regressor_model)
    
    def fit(self, X, y):
        for i in range(1, self.number_of_classes):
            y_binary = (y >= i).astype(int)
            self.models[i].fit(X, y_binary)
    
    def predict(self, X):
        preds = []
        for i in range(1, self.number_of_classes):
            pred = self.models[i].predict(X)
            preds.append(pred)
        
        preds = np.array(preds).T  # Shape: (n_samples, number_of_classes - 1)
        
        # Convert cumulative probabilities to class probabilities
        n_samples = preds.shape[0]
        class_probs = np.zeros((n_samples, self.number_of_classes))
        
        # P(y=1) = P(y>=1)
        class_probs[:, 0] = preds[:, 0]
        
        # P(y=i) = P(y>=i) - P(y>=i-1) for i=2,...,4
        for i in range(1, self.number_of_classes - 1):
            class_probs[:, i] = preds[:, i] - preds[:, i-1]
        
        # P(y=5) = 1 - P(y>=5)
        class_probs[:, -1] = 1 - preds[:, -1]
        
        return class_probs