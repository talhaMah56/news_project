import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
from typing import Tuple
import numpy as np

def fit_prophet_model(train_df: pd.DataFrame, test_df: pd.DataFrame, verbose: bool = False) -> Tuple[Prophet, pd.DataFrame]:
    """Fit a Prophet model to the time series data and plot results.

    Args:
        train_df: DataFrame with 'Timestamp' and 'Frequency' columns for training.
        test_df: DataFrame with 'Timestamp' and 'Frequency' columns for testing.
        verbose: If True, show plots; otherwise, save them to files.

    Returns:
        A tuple containing the fitted Prophet model and forecast DataFrame.
    """
    # Rename columns to match Prophet's expected format
    train_df = train_df.rename(columns={'Timestamp': 'ds', 'Frequency': 'y'})
    test_df = test_df.rename(columns={'Timestamp': 'ds', 'Frequency': 'y'})

    # Initialize and fit the Prophet model on the training data
    model = Prophet()
    model.fit(train_df)

    # Generate future dataframe for the test period
    future = model.make_future_dataframe(periods=len(test_df), freq='D')
    forecast = model.predict(future)

    # Filter forecast to only include test dates
    forecast_test = forecast[forecast['ds'].isin(test_df['ds'])]

    # Plot results
    fig1 = plt.figure(figsize=(12, 6))
    plt.plot(train_df['ds'], train_df['y'], label='Train', color='green')
    plt.plot(test_df['ds'], test_df['y'], label='Test', color='blue')
    plt.plot(forecast_test['ds'], forecast_test['yhat'], label='Forecast', color='orange', linestyle='--')
    plt.fill_between(forecast_test['ds'], forecast_test['yhat_lower'], forecast_test['yhat_upper'], color='gray', alpha=0.3)
    plt.legend()
    plt.title("Prophet Forecast vs Test Data (Train + Test)")
    plt.xlabel("Date")
    plt.ylabel("Value")

    fig2 = model.plot_components(forecast)

    # Show or save plots
    if verbose:
        fig1.show()
        fig2.show()
    else:
        os.makedirs('images', exist_ok=True)
        fig1.savefig('images/prophet_test_forecast.png')
        fig2.savefig('images/prophet_components.png')

    plt.close()
    return model, forecast
