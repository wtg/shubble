"""LSTM model for time series forecasting."""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, List, Union, Any
from tqdm import tqdm


class LSTMNet(nn.Module):
    """
    PyTorch LSTM Neural Network architecture.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.0):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.dropout_layer = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]       
        out = self.dropout_layer(out) 
        out = self.fc(out)            
        return out


class LSTMModel:
    """
    LSTM Model wrapper with scikit-learn style interface.
    """

    def __init__(
        self, 
        input_size: int = 1, 
        hidden_size: int = 50, 
        num_layers: int = 1, 
        output_size: int = 1,
        dropout: float = 0.0,
        learning_rate: float = 0.001,
        weight_decay: float =1e-5,
        batch_size: int = 32,
        epochs: int = 100,
        device: str = "cpu"
    ):
        """
        Initialize LSTM model with customizable hyperparameters.

        Args:
            input_size: Number of expected features in the input x
            hidden_size: Number of features in the hidden state h
            num_layers: Number of recurrent layers
            output_size: Number of output features
            dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of epochs to train
            device: 'cpu' or 'cuda'
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        
        self.model = LSTMNet(input_size, hidden_size, num_layers, output_size, dropout).to(self.device)
        self.criterion = nn.HuberLoss(delta=1.0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.history = []
        self.scaler: Optional[Any] = None
        self.grad_clip_max_norm = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        """
        Fit LSTM model to the data.

        Args:
            X: Input data of shape (num_samples, seq_length, input_size) or (num_samples, seq_length)
            y: Target data of shape (num_samples, output_size) or (num_samples,)
            verbose: Whether to print progress
            
        Returns:
            self
        """
        # Ensure data is correct shape and tensor
        if X.ndim == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        self.history = []
        
        epoch_iterator = range(self.epochs)
        if verbose:
            epoch_iterator = tqdm(epoch_iterator, desc="Training LSTM")
        
        for epoch in epoch_iterator:
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_max_norm)
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            self.history.append(avg_loss)
            
            if verbose:
                epoch_iterator.set_postfix(loss=f"{avg_loss:.6f}")
                
        return self

    def set_scaler(self, scaler: Any) -> "LSTMModel":
        """Set the input StandardScaler used for normalization at train and predict time."""
        self.scaler = scaler
        return self

    def _apply_scaler(self, X: np.ndarray) -> np.ndarray:
        """Apply scaler to X (n_samples, seq_len, n_features). Reshape, transform, reshape back."""
        if self.scaler is None:
            return X
        n_samples, seq_len, n_features = X.shape
        X_2d = X.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_2d)
        return X_scaled.reshape(n_samples, seq_len, n_features)

    def predict(self, X: np.ndarray):
        """
        Generate predictions.

        Args:
            X: Input data of shape (num_samples, seq_length, input_size)

        Returns:
            Array of predictions
        """
        if X.ndim == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)

        if self.scaler is not None:
            X = self._apply_scaler(X.astype(np.float64)).astype(np.float32)
            
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
            
        return predictions.cpu().numpy()

    def save(self, path: str, scaler_path: Optional[str] = None):
        """Save model state dict. If scaler_path is set and model has a scaler, save scaler too."""
        torch.save(self.model.state_dict(), path)
        if scaler_path is not None and self.scaler is not None:
            import pickle
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)

    def load(self, path: str, scaler_path: Optional[str] = None):
        """Load model state dict. If scaler_path is set and file exists, load scaler and set on model."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        if scaler_path is not None:
            from pathlib import Path
            if Path(scaler_path).exists():
                import pickle
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
