import sys
import os
import pickle
import numpy as np
import torch
import torch.nn as nn

from Models.Saint.model import SAINT


# Add paths to modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Data'))

class SaintWrapper:
    """
    Wrapper to make SAINT (PyTorch model) sklearn-compatible.
    
    Parameters:
    -----------
    num_categorical : int
        Number of categorical features
    num_continuous : int
        Number of continuous features
    num_classes : int
        Number of output classes
    cat_dims : list
        Dimensions (cardinality) of each categorical feature
    device : str
        Device to use ('cuda' or 'cpu')
    """
    
    def __init__(self, num_categorical, num_continuous, num_classes, cat_dims, device='cpu', task='classification',attention_type='col'):
        self.num_categorical = num_categorical
        self.num_continuous = num_continuous
        self.num_classes = num_classes
        self.cat_dims = cat_dims
        self.device = device
        self.task = task
        self.is_fitted = False
        self.attention_type=attention_type
        
        if self.task not in ('classification', 'regression'):
            raise ValueError("task must be 'classification' or 'regression'")

        # Determine output dimension
        if self.task == 'regression':
            # For regression, output dimension equals num_classes (multi-output regression)
            # Useful for predicting class-wise residuals in boosting
            out_dim = num_classes if (num_classes is not None) else 1
        else:
            out_dim = num_classes if (num_classes is not None) else 1
        
        # Initialize model
        self.model = SAINT(
            categories=tuple(cat_dims),
            num_continuous=num_continuous,
            dim=32,
            depth=3,
            heads=3,
            dim_head=16,
            dim_out=out_dim,
            mlp_hidden_mults=(4, 2),
            mlp_act=None,
            num_special_tokens=1,
            attn_dropout=0.1,
            ff_dropout=0.1,
            attentiontype=self.attention_type
        ).to(device)
        
        self.optimizer = None
        # Set loss / criterion depending on task
        if self.task == 'regression':
            self.criterion = nn.MSELoss()
        else:
            if num_classes == 1:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.CrossEntropyLoss()

    """    
    def _create_embeddings(self, X_cat, X_cont):
        
        Create embeddings for categorical and continuous features.
        
        Parameters:
        -----------
        X_cat : torch.Tensor
            Categorical features (batch_size, num_categorical)
        X_cont : torch.Tensor
            Continuous features (batch_size, num_continuous)
            
        Returns:
        --------
        tuple : (X_cat_enc, X_cont_enc) encoded features with proper shapes
        
        batch_size = X_cat.shape[0]
        
        # Create categorical embeddings using embedding layer
        # Output shape: (batch_size, num_categorical, dim)
        X_cat_enc = self.model.embeds(X_cat.long())
        
        # For continuous features, process through simple MLPs if available
        X_cont_enc_list = []
        if hasattr(self.model, 'simple_MLP') and len(self.model.simple_MLP) > 0:
            num_continuous = X_cont.shape[1]
            for i in range(min(num_continuous, len(self.model.simple_MLP))):
                cont_feature = X_cont[:, i:i+1]  # Shape: (batch_size, 1)
                cont_embedding = self.model.simple_MLP[i](cont_feature)  # Shape: (batch_size, dim)
                X_cont_enc_list.append(cont_embedding)
            
            # Stack embeddings: (batch_size, num_continuous, dim)
            X_cont_enc = torch.stack(X_cont_enc_list, dim=1)
        else:
            # If no MLP available, just expand dims and keep as is
            # Shape: (batch_size, num_continuous, 1) or handle differently
            X_cont_enc = X_cont.unsqueeze(-1)  # (batch_size, num_continuous, 1)
        
        return X_cat_enc, X_cont_enc
       """
    def fit(self, X, y,  X_val=None, y_val=None, epochs=20, batch_size=256, learning_rate=0.0005, val_metric_fn=None):
        """
        Train the Saint model.
        
        Parameters:
        -----------
        X : np.ndarray
            Input features (combined categorical and continuous)
        y : np.ndarray
            Target labels
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate for optimizer
        X_val : np.ndarray, optional
            Validation features for epoch-wise evaluation
        y_val : np.ndarray, optional
            Validation targets
        val_metric_fn : callable, optional
            Function that takes (y_true, y_pred) and returns a metric value
        """
        # Setup optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        if self.task == 'regression':
            y_tensor = torch.FloatTensor(y).to(self.device)
            if y_tensor.dim() == 1:
                y_tensor = y_tensor.unsqueeze(1)
        elif self.task == 'classification' and self.num_classes == 1:
            # Binary classification with single logit
            y_tensor = torch.FloatTensor(y).to(self.device)
            if y_tensor.dim() == 1:
                y_tensor = y_tensor.unsqueeze(1)
        else:
            y_tensor = torch.LongTensor(y).to(self.device)
        
        # Separate categorical and continuous features
        X_cat = X_tensor[:, :self.num_categorical].long()
        X_cont = X_tensor[:, self.num_categorical:]
        
        # Training loop
        self.model.train()
        num_batches = (len(X) + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(X))
                
                # Get batch
                batch_cat = X_cat[start_idx:end_idx]
                batch_cont = X_cont[start_idx:end_idx]
                batch_y = y_tensor[start_idx:end_idx]
                
                # Create embeddings for this batch (fresh graph for each batch)
                batch_cat_enc, batch_cont_enc = self._create_embeddings(batch_cat, batch_cont)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_cat, batch_cont, batch_cat_enc, batch_cont_enc)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / num_batches
            
            # Print loss and validation metric every 4 epochs or at the end
            if ((epoch + 1) % 4 == 0) or (epoch == epochs - 1):
                msg = f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}"
                
                # Compute validation metric if provided
                if X_val is not None and y_val is not None and val_metric_fn is not None:
                    val_pred = self.predict(X_val)
                    val_metric = val_metric_fn(y_val, val_pred)
                    msg += f", Val Metric: {val_metric:.4f}"
                
                print(msg)
        
        self.is_fitted = True
        
    def predict(self, X):
        """
        Make predictions on new data.
        For classification returns class labels; for regression returns continuous predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        
        with torch.no_grad():
            # Convert to tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            # Separate categorical and continuous features
            X_cat = X_tensor[:, :self.num_categorical].long()
            X_cont = X_tensor[:, self.num_categorical:]
            
            # Create embeddings for categorical features
            X_cat_enc, X_cont_enc = self._create_embeddings(X_cat, X_cont)
            
            # Forward pass
            outputs = self.model(X_cat, X_cont, X_cat_enc, X_cont_enc)

            if self.task == 'regression':
                # For regression: return predictions with shape (n_samples,) or (n_samples, num_outputs)
                predictions = outputs.cpu().numpy()
                # Only squeeze if single output (num_classes=1)
                if predictions.shape[1] == 1:
                    predictions = predictions.squeeze(-1)
            else:
                if self.num_classes == 1:
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    predictions = (probs >= 0.5).astype(int).squeeze()
                else:
                    predictions = outputs.argmax(dim=1).cpu().numpy()
        
        return predictions
    
    def predict_proba(self, X):
        """
        Get probability predictions (classification only).
        
        For multi-class returns shape (n_samples, n_classes).
        For binary (num_classes==1) returns shape (n_samples, 2) columns [prob_negative, prob_positive].
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if self.task != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
        
        self.model.eval()
        
        with torch.no_grad():
            # Convert to tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            # Separate categorical and continuous features
            X_cat = X_tensor[:, :self.num_categorical].long()
            X_cont = X_tensor[:, self.num_categorical:]
            
            # Create embeddings for categorical features
            X_cat_enc, X_cont_enc = self._create_embeddings(X_cat, X_cont)
            
            # Forward pass
            outputs = self.model(X_cat, X_cont, X_cat_enc, X_cont_enc)

            if self.num_classes == 1:
                probs_pos = torch.sigmoid(outputs).cpu().numpy().squeeze(-1)
                probs = np.vstack([1 - probs_pos, probs_pos]).T
            else:
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
        
        return probs
    
    @property
    def n_features_in_(self):
        return self.num_categorical + self.num_continuous
    
    @property
    def n_classes_(self):
        return self.num_classes if self.task == 'classification' else None