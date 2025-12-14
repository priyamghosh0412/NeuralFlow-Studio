import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error, accuracy_score, precision_score, recall_score

class Trainer:
    def __init__(self, builder, epochs=5, batch_size=32):
        self.builder = builder
        self.epochs = epochs
        self.batch_size = batch_size

    def train_and_evaluate(self, architecture, X_train, y_train, X_val, y_val):
        """
        Builds, trains, and evaluates the model.
        Returns:
            reward: Float (Primary metric)
            metrics: Dict of all metrics
        """
        model = self.builder.build_model(architecture)
        
        if model is None:
            # Invalid architecture (e.g. shape mismatch)
            return -1.0, {}

        try:
            # Train
            # We use a small number of epochs for NAS speed
            print(f"DEBUG: Starting model.fit for {self.epochs} epochs...")
            history = model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_val, y_val),
                verbose=0
            )
            print("DEBUG: model.fit completed.")
            
            # Evaluate
            y_pred = model.predict(X_val, verbose=0)
            
            metrics = {}
            reward = 0.0
            
            if self.builder.problem_type == 'classification':
                # For multiclass: y_pred is (batch, num_classes), y_val is (batch,) or (batch, 1)
                # We need to convert predictions to class indices
                if y_pred.shape[1] > 1:
                    # Multiclass: take argmax of predictions
                    y_pred_indices = np.argmax(y_pred, axis=1)
                else:
                    # Binary: threshold at 0.5
                    y_pred_indices = (y_pred.flatten() > 0.5).astype(int)
                
                # Ensure y_val is 1D
                if y_val.ndim > 1:
                    y_val_indices = y_val.flatten()
                else:
                    y_val_indices = y_val

                acc = accuracy_score(y_val_indices, y_pred_indices)
                prec = precision_score(y_val_indices, y_pred_indices, average='weighted', zero_division=0)
                rec = recall_score(y_val_indices, y_pred_indices, average='weighted', zero_division=0)
                
                metrics = {'accuracy': acc, 'precision': prec, 'recall': rec}
                reward = acc # Primary reward
                
            elif self.builder.problem_type == 'regression':
                r2 = r2_score(y_val, y_pred)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                mape = mean_absolute_percentage_error(y_val, y_pred)
                one_minus_mape = 1 - mape
                
                metrics = {'r2': r2, 'rmse': rmse, 'mape': mape, 'one_minus_mape': one_minus_mape}
                reward = one_minus_mape # User requested 1-MAPE as main metric? Or just for threshold?
                # User said "based on rewards increase size... if main performance metric... reaches threshold".
                # Let's use 1-MAPE as reward for consistency with the goal.
                # But R2 is also good. Let's stick to user request: "main performance metric... 1-MAPE".
                
            return reward, metrics, model

        except Exception as e:
            print(f"Training failed: {e}")
            return -1.0, {}, None
