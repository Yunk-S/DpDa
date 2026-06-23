import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_deep_learning_models(self, epochs=100, batch_size=32, patience=10):
    """Train deep learning models"""
    print("\nStarting deep learning model training...")
    
    # Create model structures
    for dataset_name, (X_train, X_test, y_train, y_test) in self.data_splits.items():
        print(f"\nTraining deep learning models for {dataset_name} dataset...")
        
        # Determine task type
        is_regression = False
        if dataset_name == 'cirrhosis':  # Cirrhosis dataset is a regression task
            is_regression = True
        
        # Prepare data
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.FloatTensor(y_train.values)
        X_test_tensor = torch.FloatTensor(X_test.values)
        y_test_tensor = torch.FloatTensor(y_test.values)
        
        if not is_regression:
            # Classification task
            y_train_tensor = y_train_tensor.reshape(-1, 1)
            y_test_tensor = y_test_tensor.reshape(-1, 1)
            
            # Create DataLoader
            train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Create model
            input_dim = X_train.shape[1]
            
            # Simple DNN model
            dnn_model = self.DNN(input_dim=input_dim).to(self.device)
            dnn_optimizer = optim.Adam(dnn_model.parameters(), lr=0.001)
            
            # DNN model with attention mechanism
            attention_model = self.AttentionDNN(input_dim=input_dim).to(self.device)
            attention_optimizer = optim.Adam(attention_model.parameters(), lr=0.001)
            
            # Define binary classification loss function
            criterion = nn.BCELoss()
            
            # Train model
            best_dnn_loss = float('inf')
            best_att_loss = float('inf')
            dnn_patience_counter = 0
            att_patience_counter = 0
            
            for epoch in range(1, epochs + 1):
                dnn_model.train()
                attention_model.train()
                
                dnn_epoch_loss = 0
                att_epoch_loss = 0
                
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    # Train DNN
                    dnn_optimizer.zero_grad()
                    dnn_outputs = dnn_model(batch_X)
                    dnn_loss = criterion(dnn_outputs, batch_y)
                    dnn_loss.backward()
                    dnn_optimizer.step()
                    dnn_epoch_loss += dnn_loss.item()
                    
                    # Train Attention DNN
                    attention_optimizer.zero_grad()
                    att_outputs = attention_model(batch_X)
                    att_loss = criterion(att_outputs, batch_y)
                    att_loss.backward()
                    attention_optimizer.step()
                    att_epoch_loss += att_loss.item()
                
                dnn_epoch_loss /= len(train_loader)
                att_epoch_loss /= len(train_loader)
                
                # Print training progress
                if epoch % 10 == 0:
                    print(f"  Epoch {epoch}/{epochs}, DNN Loss: {dnn_epoch_loss:.4f}, Attention Loss: {att_epoch_loss:.4f}")
                
                # Early stopping
                if dnn_epoch_loss < best_dnn_loss:
                    best_dnn_loss = dnn_epoch_loss
                    dnn_patience_counter = 0
                else:
                    dnn_patience_counter += 1
                
                if att_epoch_loss < best_att_loss:
                    best_att_loss = att_epoch_loss
                    att_patience_counter = 0
                else:
                    att_patience_counter += 1
                
                if dnn_patience_counter >= patience and att_patience_counter >= patience:
                    print(f"  Early stopping training, no improvement: {patience} epochs")
                    break
            
            # Evaluate model
            dnn_model.eval()
            attention_model.eval()
            
            with torch.no_grad():
                dnn_outputs = dnn_model(X_test_tensor.to(self.device))
                att_outputs = attention_model(X_test_tensor.to(self.device))
                
                # Convert to numpy for evaluation
                dnn_probs = dnn_outputs.cpu().numpy()
                att_probs = att_outputs.cpu().numpy()
                
                dnn_predictions = (dnn_probs > 0.5).astype(int).flatten()
                att_predictions = (att_probs > 0.5).astype(int).flatten()
                
                y_test_np = y_test.values
                
                # Calculate evaluation metrics
                dnn_accuracy = accuracy_score(y_test_np, dnn_predictions)
                att_accuracy = accuracy_score(y_test_np, att_predictions)
                
                dnn_f1 = f1_score(y_test_np, dnn_predictions, average='weighted')
                att_f1 = f1_score(y_test_np, att_predictions, average='weighted')
                
                # Try to calculate AUC (for binary classification)
                try:
                    dnn_auc = roc_auc_score(y_test_np, dnn_probs)
                    att_auc = roc_auc_score(y_test_np, att_probs)
                    print(f"  DNN AUC: {dnn_auc:.4f}")
                    print(f"  Attention DNN AUC: {att_auc:.4f}")
                except:
                    dnn_auc = 0
                    att_auc = 0
                
                print(f"  DNN Accuracy: {dnn_accuracy:.4f}, F1: {dnn_f1:.4f}")
                print(f"  Attention DNN Accuracy: {att_accuracy:.4f}, F1: {att_f1:.4f}")
                
            # Select better performing model as teacher
            if att_f1 > dnn_f1:
                self.teacher_models[dataset_name] = attention_model
                print(f"  Selected Attention DNN as teacher model for {dataset_name} dataset")
            else:
                self.teacher_models[dataset_name] = dnn_model
                print(f"  Selected DNN as teacher model for {dataset_name} dataset")
                
            # Save both models
            self.dl_models[dataset_name] = {
                'dnn': dnn_model,
                'attention_dnn': attention_model
            }
            
        else:
            # Regression task - use MSE loss
            y_train_tensor = y_train_tensor.reshape(-1, 1)
            y_test_tensor = y_test_tensor.reshape(-1, 1)
            
            # Create DataLoader
            train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Create model - regression output without activation function
            input_dim = X_train.shape[1]
            
            # Regression DNN model
            reg_dnn_model = self.DNN(input_dim=input_dim, output_dim=1).to(self.device)
            reg_dnn_optimizer = optim.Adam(reg_dnn_model.parameters(), lr=0.001)
            
            # Regression attention DNN model  
            reg_attention_model = self.AttentionDNN(input_dim=input_dim, output_dim=1).to(self.device)
            reg_attention_optimizer = optim.Adam(reg_attention_model.parameters(), lr=0.001)
            
            # Define regression loss function
            reg_criterion = nn.MSELoss()
            
            # Train model
            best_dnn_loss = float('inf')
            best_att_loss = float('inf')
            dnn_patience_counter = 0
            att_patience_counter = 0
            
            for epoch in range(1, epochs + 1):
                reg_dnn_model.train()
                reg_attention_model.train()
                
                dnn_epoch_loss = 0
                att_epoch_loss = 0
                
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    # Train DNN
                    reg_dnn_optimizer.zero_grad()
                    dnn_outputs = reg_dnn_model(batch_X)
                    dnn_loss = reg_criterion(dnn_outputs, batch_y)
                    dnn_loss.backward()
                    reg_dnn_optimizer.step()
                    dnn_epoch_loss += dnn_loss.item()
                    
                    # Train Attention DNN
                    reg_attention_optimizer.zero_grad()
                    att_outputs = reg_attention_model(batch_X)
                    att_loss = reg_criterion(att_outputs, batch_y)
                    att_loss.backward()
                    reg_attention_optimizer.step()
                    att_epoch_loss += att_loss.item()
                
                dnn_epoch_loss /= len(train_loader)
                att_epoch_loss /= len(train_loader)
                
                # Print training progress
                if epoch % 10 == 0:
                    print(f"  Epoch {epoch}/{epochs}, DNN Loss: {dnn_epoch_loss:.4f}, Attention Loss: {att_epoch_loss:.4f}")
                
                # Early stopping
                if dnn_epoch_loss < best_dnn_loss:
                    best_dnn_loss = dnn_epoch_loss
                    dnn_patience_counter = 0
                else:
                    dnn_patience_counter += 1
                
                if att_epoch_loss < best_att_loss:
                    best_att_loss = att_epoch_loss
                    att_patience_counter = 0
                else:
                    att_patience_counter += 1
                
                if dnn_patience_counter >= patience and att_patience_counter >= patience:
                    print(f"  Early stopping training, no improvement: {patience} epochs")
                    break
            
            # Evaluate model
            reg_dnn_model.eval()
            reg_attention_model.eval()
            
            with torch.no_grad():
                dnn_outputs = reg_dnn_model(X_test_tensor.to(self.device))
                att_outputs = reg_attention_model(X_test_tensor.to(self.device))
                
                # Convert to numpy for evaluation
                dnn_preds = dnn_outputs.cpu().numpy().flatten()
                att_preds = att_outputs.cpu().numpy().flatten()
                
                y_test_np = y_test.values
                
                # Calculate regression evaluation metrics
                dnn_mse = mean_squared_error(y_test_np, dnn_preds)
                att_mse = mean_squared_error(y_test_np, att_preds)
                
                dnn_mae = mean_absolute_error(y_test_np, dnn_preds)
                att_mae = mean_absolute_error(y_test_np, att_preds)
                
                dnn_r2 = r2_score(y_test_np, dnn_preds)
                att_r2 = r2_score(y_test_np, att_preds)
                
                print(f"  DNN MSE: {dnn_mse:.4f}, MAE: {dnn_mae:.4f}, R2: {dnn_r2:.4f}")
                print(f"  Attention DNN MSE: {att_mse:.4f}, MAE: {att_mae:.4f}, R2: {att_r2:.4f}")
                
            # Select better performing model as teacher
            if att_r2 > dnn_r2:
                self.teacher_models[dataset_name] = reg_attention_model
                print(f"  Selected Attention DNN as teacher model for {dataset_name} dataset")
            else:
                self.teacher_models[dataset_name] = reg_dnn_model
                print(f"  Selected DNN as teacher model for {dataset_name} dataset")
                
            # Save both models
            self.dl_models[dataset_name] = {
                'dnn': reg_dnn_model,
                'attention_dnn': reg_attention_model
            }
    
    return self.teacher_models
