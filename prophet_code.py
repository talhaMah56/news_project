import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
from typing import Tuple
import numpy as np


def fit_prophet_model(train_df: pd.DataFrame,
                      test_df: pd.DataFrame,
                      verbose: bool = False,
                      image_suffix: str = "",
                      use_exp: bool = False) -> Tuple[Prophet, pd.DataFrame]:
    """Fit a Prophet model to the time series data and plot results.

    Args:
        train_df: DataFrame with 'Timestamp' and 'Frequency' columns for training.
        test_df: DataFrame with 'Timestamp' and 'Frequency' columns for testing.
        verbose: If True, show plots; otherwise, save them to files.
        image_suffix: Optional suffix for custom image filenames.
        use_logistic: If True, use logistic growth model instead of linear.

    Returns:
        A tuple containing the fitted Prophet model and forecast DataFrame.
    """
    # Rename columns to match Prophet's expected format
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df = train_df.rename(columns={'Timestamp': 'ds', 'Frequency': 'y'})
    test_df = test_df.rename(columns={'Timestamp': 'ds', 'Frequency': 'y'})
    train_df['cap'] = 10**3
    model = Prophet(growth='logistic' if use_exp else 'linear')

    # Initialize and fit the Prophet model
    model.fit(train_df)

    # Generate future dataframe based on test data
    future = model.make_future_dataframe(periods=len(test_df), freq='M')
    future['cap'] = 10**3

    forecast = model.predict(future)

    # Plot results
    fig1 = plt.figure(figsize=(12, 6))

    plt.plot(train_df['ds'], train_df['y'], label='Train', color='green')
    plt.plot(test_df['ds'], test_df['y'], label='Test', color='blue')
    plt.plot(forecast['ds'],
             forecast['yhat'],
             label='Forecast',
             color='orange',
             linestyle='--')
    plt.fill_between(forecast['ds'],
                     forecast['yhat_lower'],
                     forecast['yhat_upper'],
                     color='gray',
                     alpha=0.3)
    plt.legend()
    if (use_exp):
        plt.title(f"Prophet Forecast Logistic vs Test Data")
    else: 
        plt.title(f"Prophet Forecast Linear vs Test Data")
    plt.xlabel("Date")
    plt.ylabel("Value")

    fig2 = model.plot_components(forecast)

    # Show or save plots
    if verbose:
        fig1.show()
        fig2.show()
    else:
        os.makedirs('images', exist_ok=True)
        fig1.savefig(
            f'images/prophet_forecast_{"exp" if use_exp else "lin"}_{image_suffix}.png'
        )
        fig2.savefig(
            f'images/prophet_components_{"exp" if use_exp else "lin"}_{image_suffix}.png'
        )

    plt.close()
    return model, forecast
