"""ARIMA model for time series forecasting."""
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel:
    """
    ARIMA (AutoRegressive Integrated Moving Average) model wrapper.

    ARIMA(p, d, q) where:
    - p: Order of the autoregressive part
    - d: Degree of differencing
    - q: Order of the moving average part
    """

    def __init__(self, p: int = 1, d: int = 0, q: int = 1):
        """
        Initialize ARIMA model with specified orders.

        Args:
            p: AR order (number of lag observations)
            d: Differencing order (number of times to difference)
            q: MA order (size of moving average window)
        """
        self.p = p
        self.d = d
        self.q = q
        self.order = (p, d, q)
        self.model = None
        self.results = None

    def fit(self, endog: np.ndarray, **kwargs):
        """
        Fit ARIMA model to the data.

        Args:
            endog: Endogenous variable (time series data)
            **kwargs: Additional arguments passed to ARIMA.fit()

        Returns:
            ARIMAResults object
        """
        # Create model instance
        self.model = ARIMA(endog, order=self.order)

        # Fit model
        self.results = self.model.fit(**kwargs)

        return self.results

    def predict(self, n_periods: int = 1, **kwargs):
        """
        Generate predictions.

        Args:
            n_periods: Number of periods to forecast
            **kwargs: Additional arguments passed to predict()

        Returns:
            Array of predictions
        """
        if self.results is None:
            raise ValueError("Model must be fitted before making predictions")

        return self.results.forecast(steps=n_periods, **kwargs)

    def get_one_step_predictions(self):
        """
        Get one-step-ahead predictions (fitted values).

        Returns:
            Array of one-step-ahead predictions
        """
        if self.results is None:
            raise ValueError("Model must be fitted before getting predictions")

        return self.results.fittedvalues

    @property
    def aic(self):
        """Akaike Information Criterion."""
        if self.results is None:
            return None
        return self.results.aic

    @property
    def bic(self):
        """Bayesian Information Criterion."""
        if self.results is None:
            return None
        return self.results.bic

    def summary(self):
        """Get model summary."""
        if self.results is None:
            raise ValueError("Model must be fitted before getting summary")
        return self.results.summary()
