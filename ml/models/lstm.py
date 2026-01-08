"""LSTM model for time series forecasting."""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, List, Union
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
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
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
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.history = []

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
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            self.history.append(avg_loss)
            
            if verbose:
                epoch_iterator.set_postfix(loss=f"{avg_loss:.6f}")
                
        return self

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
            
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
            
        return predictions.cpu().numpy()

    def save(self, path: str):
        """Save model state dict."""
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        """Load model state dict."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
