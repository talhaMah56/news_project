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

    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(train_df)

    # Generate future dataframe based on test data
    future = test_df[['ds']].copy()
    forecast = model.predict(future)

    # No need to filter if future is based on test_df

    # Plot results
    fig1 = plt.figure(figsize=(12, 6))
    plt.plot(test_df['ds'], test_df['y'], label='Test', color='blue')
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='orange', linestyle='--')
    plt.legend()
    plt.title("Prophet Forecast vs Test Data")
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
